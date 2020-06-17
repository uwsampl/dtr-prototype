"""Utilities for validating a config object"""

def check_item(field, value, acceptable_values, conditions):
    if field in acceptable_values:
        acceptable_set = acceptable_values[field]
        if value not in acceptable_set:
            return (None,
                    "Invalid value {} for field {}".format(value, field))
    if field in conditions:
        condition, name = conditions[field]
        if not condition(value):
            return (None,
                    "{} does not meet precondition \"{}\" for field {}".format(value,
                                                                           field,
                                                                           name))
    return (value, "")

def non_negative_cond():
    return (lambda value: isinstance(value, int) and value >= 0, "must be non-negative")

def string_cond():
    return (lambda value: isinstance(value, str), "must be string")

def bool_cond():
    return (lambda value: isinstance(value, bool), "must be bool")

def check_config(config, defaults=None, acceptable_values=None, conditions=None, permit_empty=None):
    """
    Given a config object (dict), set of default values (config field -> value),
    set of acceptable values (config field -> set of permitted values; not all fields need to
    be populated), set of conditions to check (config field -> lambda that takes the value
    and returns boolean), and a set of fields that permit empty lists, returns
    1. a sanitized config object (None if there is an error)
    2. an error message to report if a condition fails or there is an invalid value
    Note that for config fields that are lists, this function will turn them into sets
    to deduplicate.
    defaults, acceptable_values, conditions, permit_empty are all empty by default
    """
    defaults = defaults if defaults is not None else {}
    acceptable_values = acceptable_values if acceptable_values is not None else {}
    conditions = conditions if conditions is not None else {}
    permit_empty = permit_empty if permit_empty is not None else {'notify'}
    ret = defaults

    for field, value in config.items():
        # for lists, validate each item and turn into a set
        if isinstance(value, list):
            if field not in permit_empty and len(value) == 0:
                return (None, "Config field {} is not permitted to be empty".format(field))

            checked_list = []
            for i in value:
                v, msg = check_item(field, i, acceptable_values, conditions)
                if v is None:
                    return (None, msg)
                checked_list.append(v)
            ret[field] = set(checked_list)
            continue

        v, msg = check_item(field, value, acceptable_values, conditions)
        if v is None:
            return (None, msg)
        ret[field] = v

    return ret, ''
