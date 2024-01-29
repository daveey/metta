import argparse

VALUES_RANGE_DELIMITER = ":"

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


