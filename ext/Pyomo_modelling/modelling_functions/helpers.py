import numpy as np
import ast
from fractions import Fraction
from functools import reduce
from collections.abc import Iterable

def split_name_and_index(component_str):
    in_single = False
    in_double = False

    for i in range(len(component_str) - 1, -1, -1):
        c = component_str[i]

        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c == "[" and not in_single and not in_double:
            # Found the outermost index
            name = component_str[:i]
            index = component_str[i + 1 : -1]
            return name.strip(), index.strip()

    return component_str.strip(), None


def get_component_by_name(model, component_str):
    """
    Retrieve a Pyomo component by name, handling:
      - quoted names
      - brackets inside names
      - tuple indices

    Examples:
      x[1]
      A[1]_integer_encoding_variable[0]
      'relation[1,2]_slack_variable'[0]
    """
    name, index = split_name_and_index(component_str)

    # Unquote name if needed
    if name and name[0] in ("'", '"'):
        name = ast.literal_eval(name)

    component = getattr(model, name, None)
    if component is None:
        return None

    if index is None:
        return component

    idx = ast.literal_eval(index)
    return component[idx]
    
def set_new_attr_name(object, prefix, suffix):
    '''Set a name for an attribute to be added to an object, the name is made sure not to coincide with another attribute.
    Inputs:
        - object: object that the attribute shall be added to.
        - prefix: str, prefix of name to be chosen.
        - suffix: str, suffix of name to be chosen.
    Output:
        - str, prefix + _ + suffix, or with more underscores
    '''
    new_attr_name_base = prefix + '_'
    while hasattr(object, new_attr_name_base + suffix):
        new_attr_name_base += '_'
    return new_attr_name_base + suffix
    
def make_QUBO_upper_triangle(Q):
    '''Transform a QUBO in array form into an upper triangular QUBO'''
    Q = np.array(Q)
    return(np.triu(Q) + np.tril(Q,-1).T)

def gcd_decimal(*args):
    '''Generalization of greatest common denominator to decimal numbers'''
    # Convert decimal numbers to fractions
    fractions = [Fraction(str(arg)).limit_denominator() for arg in args]
    
    # Calculate the common denominator
    common_denominator = 1
    for fraction in fractions:
        common_denominator = common_denominator * fraction.denominator // np.gcd(common_denominator, fraction.denominator)
    # Scale the numerators to the common denominator
    scaled_numerators = [fraction.numerator * (common_denominator // fraction.denominator) for fraction in fractions]
    
    # Calculate GCD of scaled numerators
    result_numerator = scaled_numerators[0]
    for num in scaled_numerators[1:]:
        result_numerator = np.gcd(result_numerator, num)
    
    # Return the result as a decimal
    return result_numerator / common_denominator

def mod_decimal(a, b):
    '''For mod on decimal numbers. This avoids floating point errors as are happening e.g. with 8.1 % 1.'''
    assert b, 'Choose b non-zero.'
    # Convert decimals to fractions
    fraction_a = Fraction(str(a)).limit_denominator()
    fraction_b = Fraction(str(b)).limit_denominator()
    # Calculate modulus
    result = fraction_a - (fraction_a // fraction_b) * fraction_b
    
    return float(result)

def clear_denominators(*args):
    """
    Takes arbitrary numeric inputs (ints, floats, None, or nested sequences),
    clears denominators using the least common multiple,
    then simplifies results by dividing by their overall GCD.
    'None' values are preserved and ignored in calculations.
    Returns results in the same nested structure as the input.
    """

    # Convert everything to Fraction objects, skipping None
    def to_fractions(x):
        if x is None:
            return None
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            return [to_fractions(i) for i in x]
        else:
            return Fraction(x).limit_denominator()

    # Collect all denominators recursively, ignoring None
    def collect_denominators(x):
        if x is None:
            return []
        if isinstance(x, list):
            return [d for i in x for d in collect_denominators(i)]
        else:
            return [x.denominator]

    # Scale fractions by LCD, skip None
    def rescale(x, lcd):
        if x is None:
            return None
        if isinstance(x, list):
            return [rescale(i, lcd) for i in x]
        else:
            return int(x * lcd)

    # Collect all scaled integers recursively, ignoring None
    def collect_integers(x):
        if x is None:
            return []
        if isinstance(x, list):
            return [n for i in x for n in collect_integers(i)]
        else:
            return [x]

    # Divide everything by GCD recursively, skip None
    def divide_by_gcd(x, g):
        if x is None:
            return None
        if isinstance(x, list):
            return [divide_by_gcd(i, g) for i in x]
        else:
            return x // g

    # Convert all inputs to fractions (ignoring None)
    fractions_args = [to_fractions(arg) for arg in args]

    # Find LCD of all denominators (ignoring None)
    denominators = []
    for f in fractions_args:
        denominators.extend(collect_denominators(f))
    lcd = 1
    for d in denominators:
        lcd = np.lcm(lcd, d)

    # Scale everything
    scaled_args = [rescale(f, lcd) for f in fractions_args]

    # Collect all integers for GCD, ignoring None
    integers = []
    for s in scaled_args:
        integers.extend(collect_integers(s))

    # If all inputs are None, overall_gcd should be 1 to avoid division errors
    overall_gcd = reduce(np.gcd, integers) if integers else 1

    # Divide everything by GCD
    simplified_args = [divide_by_gcd(s, overall_gcd) for s in scaled_args]

    # Return tuple if multiple args, else unwrap single argument
    return simplified_args[0] if len(simplified_args) == 1 else tuple(simplified_args)