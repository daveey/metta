import argparse
from typing import List, Callable
VALUES_RANGE_DELIMITER = ":"


def get_value_possibly_from_range(vals: List, range_selection_fn: Callable):
    """
        Args: 
            vals: a list of 1 or 2 values
            range_selection_fn: a function that selects a value based on a (low, high) range boundaries
    """
    if len(vals) == 1:
        return lambda: vals[0]
    elif len(vals) == 2:
        return lambda: range_selection_fn(vals[0], vals[1])
    else:
        raise ValueError(f"Length of values list should be at most 2. Got: {len(vals)}")
    
class PossiblyNumericRange2Number(argparse.Action):
     def __init__(self, str2numeric_cast_fn, *args, **kwargs):
        """
            Args:
                str2numeric_cast_fn - a str2numeric function like int, float
        """
        self.str2numeric_cast_fn = str2numeric_cast_fn
        super().__init__(*args, **kwargs)

     def __call__(self, parser, namespace, values, option_string=None):
        vals = [self.str2numeric_cast_fn(val) for val in values.split(VALUES_RANGE_DELIMITER)]

        if len(vals) > 2:
            raise argparse.ArgumentTypeError(f"Input length (after splitting with character {VALUES_RANGE_DELIMITER} must be 1 or 2. Got {len(vals)}")
        
        setattr(namespace, self.dest, vals)


