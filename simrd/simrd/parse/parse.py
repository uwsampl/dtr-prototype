from attr import attrs, attrib, Factory
from enum import Enum, auto
import json

from ..tensor import Operator
from .graph import *

@attrs
class Call:
    result = attrib()
    name = attrib()
    args = attrib()
    time = attrib()

@attrs
class Mutate:
    name = attrib()
    args = attrib()
    mutate = attrib()
    time = attrib()

@attrs
class Constant:
    name = attrib()

@attrs
class Release:
    name = attrib()

@attrs
class Memory:
    name = attrib()
    memory = attrib()

@attrs
class Copy:
    dst = attrib()
    src = attrib()

@attrs
class CopyFrom:
    dst = attrib()
    src = attrib()

@attrs
class Annotate:
    annotation = attrib()

Annotate.BACKWARD = 'BACKWARD'
Annotate.START = 'START'

@attrs
class Alias:
    name = attrib()
    alias = attrib()

@attrs
class Unknown:
    line = attrib()

@attrs
class ParseError:
    line = attrib()

class OutputCondition(Enum):
    REMATERIALIZE = auto()
    PREALLOCATE = auto()

def parse(line):
    try:
        j = json.loads(line)
    except Exception as e:
        return ParseError(line + str(e))
    instr = j["INSTRUCTION"]
    if instr == "CONSTANT":
        return Constant(j["NAME"])
    elif instr == "MEMORY":
        return Memory(j["NAME"], j["MEMORY"])
    elif instr == "COPY":
        return Copy(j["DST"], j["SRC"])
    elif instr == "RELEASE":
        return Release(j["NAME"])
    elif instr == "CALL":
        return Call(j["RESULT"], j["NAME"], j["ARGS"], j["TIME"])
    elif instr == "MUTATE":
        return Mutate(j["NAME"], j["ARGS"], j["MUTATE"], j["TIME"])
    elif instr == "ANNOTATE":
        return Annotate(j["ANNOTATION"])
    elif instr == "COPY_FROM":
        return CopyFrom(j["DST"], j["SRC"])
    elif instr == "ALIAS":
        return Alias(j["NAME"], j["ALIAS"])
    else:
        return Unknown(line)

def parse_file(f, start=True, out_cond=OutputCondition.REMATERIALIZE, ignore_undefined_release=True) -> Graph:
    lines = [parse(line.rstrip()) for line in f]
    errors = [x for x in lines if isinstance(x, (ParseError, Unknown))]
    if len(errors) != 0:
        print(errors)
        raise
    lines.reverse()
    if start:
        while True:
            x = lines.pop()
            if isinstance(x, Annotate) and x.annotation == Annotate.START:
                break

    tensor_map = {}
    in_bwd = False
    g = Graph()
    g.meta['outputs'] = set()
    pop = lambda: lines.pop()

    while len(lines) > 0:
        l = pop()
        if isinstance(l, Constant):
            m = pop()
            assert isinstance(m, Memory)
            assert l.name == m.name
            op, (v,) = GOp.make(
                g, tuple(), 0, (int(m.memory),), (-1,),
                GOp.CONST_NAME, (m.name,), {'bwd': in_bwd}
            )
            g.schedule.append(GCompute(op))
            g.schedule.append(GGet(v, pin=True))
            tensor_map[m.name] = v
            tensor_map[m.name].meta['_ref'] = 1
        elif isinstance(l, Copy):
            assert l.dst not in tensor_map
            tensor_map[l.dst] = tensor_map[l.src]
            tensor_map[l.dst].meta['_ref'] += 1
            g.schedule.append(GGet(tensor_map[l.src], pin=False))
        elif isinstance(l, Release):
            if l.name in tensor_map:
                tensor_map[l.name].meta['_ref'] -= 1
                g.schedule.append(GRelease(tensor_map[l.name]))
            elif not ignore_undefined_release:
                raise RuntimeError('tried to release an undefined tensor')
        elif isinstance(l, Call):
            # TODO: use backward annotation for in_bwd
            in_bwd |= 'backward' in l.name
            cnt = len(l.result)
            memory, alias = zip(*[(pop(), pop()) for _ in range(cnt)])
            for mi in range(cnt):
                assert(isinstance(memory[mi], Memory))
                assert(l.result[mi] == memory[mi].name)
                assert(isinstance(alias[mi], Alias))
                assert(l.result[mi] == alias[mi].name)
            memory = list(memory)
            alias = list(alias)
            for i in range(cnt):
                alias[i] = int(alias[i].alias)
                memory[i] = 0 if alias[i] != -1 else int(memory[i].memory)
            args = tuple([tensor_map[x] for x in l.args])
            op, res = GOp.make(
                g, args, int(l.time), tuple(memory), tuple(alias), 
                l.name, tuple(l.result), {'bwd': in_bwd}
            )
            for i in range(cnt):
                assert l.result[i] not in tensor_map
                res[i].meta['_ref'] = 1
                tensor_map[l.result[i]] = res[i]
            g.schedule.append(GCompute(op))
        elif isinstance(l, Mutate):
            in_bwd |= 'backward' in l.name
            args = tuple([tensor_map[x] for x in l.args])
            memory = tuple([tensor_map[l.args[m]].storage_size for m in l.mutate])
            alias = tuple([-1 for _ in l.mutate])
            mut_names = tuple([args[m].name + '$' for m in l.mutate])
            op, res = GOp.make(g, args, int(l.time), memory, alias, l.name, mut_names, {'bwd': in_bwd})
            g.schedule.append(GCompute(op))
            for i, m in enumerate(l.mutate):
                old_tensor = tensor_map[l.args[m]]
                tensor_map[l.args[m]] = res[i]
                res[i].meta['_ref'] = 1
                old_tensor.meta['_ref'] -= 1
                g.schedule.append(GRelease(old_tensor))
        elif isinstance(l, Annotate):
            if l.annotation == Annotate.BACKWARD:
                in_bwd = True
        elif isinstance(l, CopyFrom):
            old_tensor = tensor_map[l.dst]
            tensor_map[l.dst] = tensor_map[l.src]
            old_tensor.meta['_ref'] -= 1
            tensor_map[l.src].meta['_ref'] += 1
            g.schedule.append(GRelease(old_tensor))
            g.schedule.append(GGet(tensor_map[l.src], pin=False))
        else:
            print(l)
            print(len(lines))
            raise

    outputs = set()
    for name in tensor_map:
        t = tensor_map[name]
        if t.meta['_ref'] > 0:
            storage_name = t.alias().name if t.alias() else t.name
            if out_cond == OutputCondition.REMATERIALIZE:
                g.schedule.append(GGet(t, pin=True))
            elif out_cond == OutputCondition.PREALLOCATE:
                if storage_name not in outputs:
                    g.meta['output_ram'] = g.meta.get('output_ram', 0) + t.storage_size
            else:
                raise RuntimeError('Unsupported output condition: {}'.format(out_cond))
            outputs.add(storage_name)
        g.meta['outputs'] = outputs

    return g
