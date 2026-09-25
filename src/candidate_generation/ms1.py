"""This file encompassess all the methods required for candidate generation of ms1 (isobars included) from a test dataset;
    columns needed : adduct, precursor m/z

"""


from ast import parse
import re
import numpy as np
import pandas as pd

from instrument_aware_tolerance import INSTRUMENT_ACCURACY_PPM

# print(df.columns)

ELEMENT_MASSES = {
    # Common organic elements
    "H": 1.00782503223,
    "C": 12.00000000000,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "P": 30.97376199842,
    "S": 31.97207117440,

    # Metals
    "Li": 7.0160034366,
    "Na": 22.9897692820,
    "K": 38.9637064864,
    "Ca": 39.9625908630,

    # Halogens
    "Cl": 34.9688526820,
    "Br": 78.9183376,

    # Other elements
    "I": 126.904468,
    "Si": 27.97692653465,
    "Se": 79.9165218,
    "As": 74.92159457,
    "B": 11.00930536,
    "Sn": 119.9022026,
}

ELECTRON_MASS = 0.000548579909065

def parse_adduct(adduct: str) -> dict:
    """Converts the adduct to a dictionary"""
    if not isinstance(adduct, str):
        raise TypeError("Adduct must be a string")

    # CATION BYPASS
    if adduct.lower().startswith("[cat]"):
        charge_match = re.search(r"\](\d*)([+-])", adduct)
        if charge_match:
            num, sign = charge_match.groups()
            magnitude = int(num) if num else 1
            charge = magnitude if sign == "+" else -magnitude
        else:
            charge = 1 
    
        return {
            "adduct": adduct,
            "charge": charge,
            "multiplier": 1,
            "ionizer": None  
        }

    pattern = r"\[(?P<multiplier>\d*)M(?P<ion_sign>[+-])(?P<ion>[^\]]+)\](?P<charge>\d*)(?P<charge_sign>[+-])"

    match = re.fullmatch(pattern, adduct)
    if not match:
        raise ValueError(f"Unsupported adduct format: {adduct}")


    mult_str = match.group("multiplier")
    multiplier = int(mult_str) if mult_str else 1

    charge_num = int(match.group("charge")) if match.group("charge") else 1
    charge_sign = match.group("charge_sign")
    charge = charge_num if charge_sign == '+' else -charge_num

    return {
        "adduct": adduct,
        "charge": charge,
        "multiplier": multiplier,
        "signed_ion" : match.group("ion_sign") + match.group("ion")
    }

def parse_ionizer(formula: str) -> dict:
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")

    formula = formula.strip()

    if not formula:
        raise ValueError("formula cannot be empty")

    composition = {}

    # Every component is optionally preceded by + or -
    components = re.findall(r"([+-]?)(\d*[A-Z][A-Za-z0-9]*)", formula)

    if not components:
        raise ValueError(f"Invalid ionizer formula: {formula}")

    # Make sure the entire string was consumed
    reconstructed = "".join(sign + component for sign, component in components)

    if reconstructed != formula:
        raise ValueError(f"Invalid ionizer formula: {formula}")

    for sign, component in components:

        # Optional multiplier before the formula
        multiplier_match = re.match(r"(\d*)", component)
        multiplier = int(multiplier_match.group(1)) if multiplier_match.group(1) else 1

        component_formula = component[len(multiplier_match.group(1)):]

        # Parse elements inside component
        position = 0

        while position < len(component_formula):

            match = re.match(
                r"([A-Z][a-z]?)(\d*)",
                component_formula[position:]
            )

            if not match:
                raise ValueError(
                    f"Invalid chemical formula: {component_formula}"
                )

            element = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1

            if element not in ELEMENT_MASSES:
                raise ValueError(
                    f"Unsupported element '{element}' in '{formula}'"
                )

            sign_value = -1 if sign == "-" else 1

            composition[element] = (
                composition.get(element, 0)
                + sign_value * multiplier * count
            )

            position += len(match.group(0))

    return composition



def calculate_mass_shift(composition:dict) -> float:
    mass = 0.0
    for key,value in composition.items():
        if key not in ELEMENT_MASSES:
            raise TypeError("Element not found")
        mass += ELEMENT_MASSES[key] * value

    return mass

def calculate_neutral_mass(precursor_mz:float,z:int,multiplier:int,mass_shift:float) -> float:
    

    nm = (precursor_mz * abs(z) - mass_shift + z * ELECTRON_MASS) / multiplier

    return nm


def append_neutral_mass(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, parse that row's adduct, compute its mass
    shift, and compute its neutral mass. Attach the result
    as a 'neutral_mass' column

    Requires 'adduct' and 'precursor_mz' columns.
    """
    if "adduct" not in df.columns:
        raise KeyError("adduct column must be present")

    if "precursor_mz" not in df.columns:
        raise KeyError("precursor_mz column must be present")


    df = df.copy()
    df["neutral_mass"] = [
        mass_for_each_row(a, p)
        for a, p in zip(df["adduct"], df["precursor_mz"])
    ]
    return df

def mass_for_each_row(adduct, precursor_mz):
    parsed = parse_adduct(adduct)
    if parsed.get("signed_ion") is None:
        # Cation: no modification to apply.
        mass_shift = 0.0
    else:
        ion = parse_ionizer(parsed["signed_ion"])
        mass_shift = calculate_mass_shift(ion)
    return calculate_neutral_mass(
        precursor_mz,
        parsed["charge"],
        parsed["multiplier"],
        mass_shift,
    )

def data_frame_modifier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by 'neutral_mass' ascending and set it as the index.
    Assumes the 'neutral_mass' column already exists.
    """
    if "neutral_mass" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'neutral_mass' column; "
            "call compute_neutral_masses first"
        )
    df = df.copy()
    df = df.sort_values("neutral_mass", kind="mergesort")
    
    return df.set_index("neutral_mass", drop=False)


def find_candidates(
    sorted_df: pd.DataFrame,
    query_mass: float,
    tolerance_ppm: float = 10.0,
) -> pd.DataFrame:
    """Numeric window search."""
    masses = sorted_df.index.to_numpy()
    delta = tolerance_ppm * query_mass / 1_000_000
    lo = np.searchsorted(masses, query_mass - delta, side="left")
    hi = np.searchsorted(masses, query_mass + delta, side="right")
    return sorted_df.iloc[lo:hi]


def find_candidates_by_instrument(
    sorted_df: pd.DataFrame,
    query_mass: float,
    instrument_name: str,
    coverage_factor: float = 1.0,
) -> pd.DataFrame:
    """Look up the instrument's tolerance, then delegate."""
    rated = INSTRUMENT_ACCURACY_PPM.get(instrument_name, 30.0)
    tolerance = rated * coverage_factor
    return find_candidates(sorted_df, query_mass, tolerance)



if __name__ == "__main__":

    data = {
        "adduct": ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"],
        "precursor_mz":[
            101.0073,  # [M+H]+
            118.0338,  # [M+NH4]+
            82.9967,   # [M-H2O+H]+
            64.9861,   # [M-2H2O+H]+
            122.9892,  # [M+Na]+
            138.9632,  # [M+K]+
            98.9927,   # [M-H]-
            80.9821,   # [M-H2O-H]-
            144.9982,  # [M+CH2O2-H]-
            134.9694,  # [M+Cl]-
        ]
    }

    df = pd.DataFrame(data)

    appended_df = append_neutral_mass(df)

    pd.set_option('display.max_columns',50)
    pd.set_option('display.max_rows',50)

    print(appended_df)

    print ("SORTED")
    sorted_df = data_frame_modifier(appended_df)
    print(sorted_df)

    candidates = find_candidates(sorted_df,100.00)


    print ("candidates")
    print(candidates)