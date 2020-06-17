from attr import attrs, attrib, Factory
import json

import simrd
from simrd.runtime import Operator

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

def parse_file(f, start_annot=False, allow_undefined_release=True, pin_live=True):
    lines = [parse(line.rstrip()) for line in f]
    errors = [x for x in lines if isinstance(x, (ParseError, Unknown))]
    if len(errors) != 0:
        print(errors)
        raise
    lines.reverse()
    if start_annot:
        while True:
            x = lines.pop()
            if isinstance(x, Annotate) and x.annotation == 'START':
                break
    def closure(rt):
        lines_copy = lines.copy()
        tensor_map = {}
        def pop():
            x = lines_copy.pop()
            return x
        while len(lines_copy) > 0:
            l = pop()
            if isinstance(l, Constant):
                m = pop()
                assert isinstance(m, Memory)
                assert l.name == m.name
                op = Operator(0, (int(m.memory),), (-1,), "constant")
                v = rt.compute([], op, names=(m.name,))[0]
                rt.pin(v)
                tensor_map[m.name] = v
            elif isinstance(l, Copy):
                assert l.dst not in tensor_map
                tensor_map[l.dst] = rt.get(tensor_map[l.src])
            elif isinstance(l, Release):
                if l.name in tensor_map:
                    rt.release(tensor_map[l.name])
                elif not allow_undefined_release:
                    raise RuntimeError('tried to release an undefined tensor')
            elif isinstance(l, Call):
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
                op = Operator(int(l.time), tuple(memory), tuple(alias), l.name)
                res = rt.compute([tensor_map[a] for a in l.args], op, names=tuple(l.result))
                for i in range(cnt):
                    assert l.result[i] not in tensor_map
                    tensor_map[l.result[i]] = res[i]
            elif isinstance(l, Mutate):
                memory = tuple([tensor_map[l.args[m]].storage.size for m in l.mutate])
                op = Operator(int(l.time), memory, tuple([-1 for _ in memory]), l.name)
                mut_names = tuple([tensor_map[l.args[m]].name + '@' for m in l.mutate])
                res = rt.compute([tensor_map[a] for a in l.args], op, names=mut_names)
                for i, m in enumerate(l.mutate):
                    rt.release(tensor_map[l.args[m]])
                    tensor_map[l.args[m]] = res[i]
            elif isinstance(l, Annotate):
                pass # Annotation does nothing.
            elif isinstance(l, CopyFrom):
                rt.release(tensor_map[l.dst])
                tensor_map[l.dst] = rt.get(tensor_map[l.src])
            else:
                print(l)
                print(len(lines))
                raise
        if pin_live:
            for name, t in tensor_map.items():
                if t.meta['ref_ext'] > 0:
                    if not t.defined:
                        rt.rematerialize(t)
                    assert t.defined
                    rt.pin(t)

    return closure
