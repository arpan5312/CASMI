"""This file encompassess all the methods required for candidate generation (isobars included) from a test dataset;
    columns needed : adduct, precursor m/z

"""


from operator import mul
import re

import pandas as pd
test_path = "data/test.parquet"

df = pd.read_parquet(test_path,engine="pyarrow",columns=["adduct","precursor_mz"])

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
    """converts the adduct to a dictionary"""
    if not isinstance(adduct,str):
        raise TypeError("Adduct must be a string")

    #CATION BYPASS
    
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
            
        }

           
    
    #NORMAL ADDUCT PARSING
    match = re.fullmatch(r"\[(?P<body>.+)\](?P<charge>\d*)(?P<sign>[+-])",adduct)
    if not match:
        raise ValueError(f"Unsupported adduct format: {adduct}")

    ionizer = re.fullmatch(
    r"\[M[+-](?P<ion>[^\]]+)\](?P<charge>\d*)(?P<sign>[+-])",
    adduct
)
    
    body = match.group("body")
    charge_num = int(match.group("charge")) if match.group("charge") else 1
    sign = match.group("sign")

    charge = charge_num if sign == '+' else -charge_num
    m = re.match(r"(\d*)M", body)
    multiplier = int(m.group(1)) if m and m.group(1) else 1

    return{
        "adduct": adduct,
        "charge": charge,
        "multiplier": multiplier,
        "ionizer":ionizer.group("ion")        
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
    """formula: NM = (m/z * |z| - mass_shift)/multiplier"""

    nm = (precursor_mz * abs(z) - mass_shift + z * ELECTRON_MASS) / multiplier

    return nm



if __name__ == "__main__":
    # Test neutral-mass calculation with a custom precursor m/z.
    adduct = "[M+H]+"
    precursor_mz = 130.0 + ELEMENT_MASSES["H"]

    parsed = parse_adduct(adduct)
    composition = parse_ionizer(parsed["ionizer"])
    mass_shift = calculate_mass_shift(composition)
    neutral_mass = calculate_neutral_mass(
        precursor_mz,
        parsed["charge"],
        parsed["multiplier"],
        mass_shift,
    )

    #assert abs(neutral_mass - 100.0) < 1e-9
    print(f"neutral mass of {adduct}: {neutral_mass}")