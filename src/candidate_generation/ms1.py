"""This file encompassess all the methods required for candidate generation of ms1 (isobars included) from a test dataset;
    columns needed : adduct, precursor m/z

"""


import re
import numpy as np
import pandas as pd

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



def data_frame_modifier(df:pd.DataFrame,neutral_mass:float) -> pd.DataFrame:
    

    if len(df) > 1:
        raise ValueError(
            "data_frame_modifier expects a single-row dataframe "
            "when neutral_mass is a scalar"
        )
    df = df.copy()
    df["neutral_mass"] = neutral_mass
    df = df.sort_values("neutral_mass", kind="mergesort")
    df = df.set_index("neutral_mass", drop=False)
    return df


def find_candidates(sorted_df:pd.DataFrame,query_mass:float,tolerance_ppm=float(10.0)) -> pd.DataFrame:

    masses = sorted_df.index.to_numpy()

    delta = tolerance_ppm*query_mass/1000000

    low = query_mass-delta
    high = query_mass+delta

    lo = np.searchsorted(masses,low,side="left")
    hi = np.searchsorted(masses,high,side="right")

    
    return sorted_df.iloc[lo:hi]



if __name__ == "__main__":

    # ========================================================
    # TEST 1 — glucose self-test for find_candidates
    # ========================================================
    #
    # Build three entries: one 0.01 Da below glucose, one at
    # glucose, one 0.01 Da above. Search with a 1 ppm window.
    # Only the middle entry should be returned, because 0.01 Da
    # at 180 Da is ~55 ppm, far outside 1 ppm.

    glucose_mass = (
        6 * ELEMENT_MASSES["C"]
        + 12 * ELEMENT_MASSES["H"]
        + 6 * ELEMENT_MASSES["O"]
    )

    glucose_df = pd.DataFrame(
        {
            "name": ["lower", "glucose", "upper"],
            "neutral_mass": [
                glucose_mass - 0.01,
                glucose_mass,
                glucose_mass + 0.01,
            ],
        }
    ).sort_values("neutral_mass").set_index("neutral_mass", drop=False)

    print()
    print("=" * 78)
    print("TEST 1 — glucose self-test")
    print("=" * 78)
    print()
    print(f"glucose neutral mass: {glucose_mass:.6f}")
    print()
    print("Library:")
    print(glucose_df.to_string())
    print()

    glucose_candidates = find_candidates(
        glucose_df, glucose_mass, tolerance_ppm=1.0
    )
    print("Search with tolerance 1 ppm:")
    print(glucose_candidates.to_string())
    print()
    assert list(glucose_candidates["name"]) == ["glucose"], (
        f"Expected ['glucose'], got {list(glucose_candidates['name'])}"
    )
    print("[PASS] only 'glucose' returned at 1 ppm")
    print()

    # ========================================================
    # TEST 2 — widen the window to catch the neighbours
    # ========================================================

    # 0.01 Da at 180 Da is ~55 ppm. A 100 ppm window should
    # include all three entries.

    glucose_candidates_wide = find_candidates(
        glucose_df, glucose_mass, tolerance_ppm=100.0
    )
    print("Search with tolerance 100 ppm:")
    print(glucose_candidates_wide.to_string())
    print()
    assert len(glucose_candidates_wide) == 3, (
        f"Expected 3 rows at 100 ppm, got {len(glucose_candidates_wide)}"
    )
    print("[PASS] all three entries returned at 100 ppm")
    print()

    # ========================================================
    # TEST 3 — empty window
    # ========================================================

    # Query a mass with nothing nearby.

    empty_result = find_candidates(
        glucose_df, 500.0, tolerance_ppm=10.0
    )
    print("Search for 500.0 at 10 ppm:")
    print(empty_result.to_string() if len(empty_result) else "(empty)")
    print()
    assert len(empty_result) == 0, (
        f"Expected 0 rows, got {len(empty_result)}"
    )
    print("[PASS] empty result returned cleanly")
    print()

    # ========================================================
    # TEST 4 — boundary inclusion
    # ========================================================
    #
    # Place an entry exactly at the lower bound of the window
    # and confirm it is included. This tests the side="left"
    # on the lower bound.

    test_mass = 100.0
    lower_bound = test_mass * (1 - 10.0 / 1_000_000)

    boundary_df = pd.DataFrame(
        {
            "name": ["at_lower_bound", "at_query", "at_upper_bound"],
            "neutral_mass": [
                lower_bound,
                test_mass,
                test_mass * (1 + 10.0 / 1_000_000),
            ],
        }
    ).sort_values("neutral_mass").set_index("neutral_mass", drop=False)

    boundary_result = find_candidates(boundary_df, test_mass, tolerance_ppm=10.0)

    print("Boundary test at 100.0, 10 ppm:")
    print(boundary_df.to_string())
    print()
    print("Result:")
    print(boundary_result.to_string())
    print()
    assert len(boundary_result) == 3, (
        f"Expected 3 rows (both bounds inclusive), got {len(boundary_result)}"
    )
    print("[PASS] both boundary entries included (closed interval)")
    print()

    # ========================================================
    # TEST 5 — neutral-mass calculation (unchanged)
    # ========================================================

    adduct = "[M+H]+"
    precursor_mz = 130.0 + ELEMENT_MASSES["H"] - ELECTRON_MASS

    parsed = parse_adduct(adduct)
    composition = parse_ionizer(parsed["ionizer"])
    mass_shift = calculate_mass_shift(composition)
    neutral_mass = calculate_neutral_mass(
        precursor_mz,
        parsed["charge"],
        parsed["multiplier"],
        mass_shift,
    )

    print("=" * 78)
    print("TEST 5 — neutral-mass calculation")
    print("=" * 78)
    print()
    print(f"adduct         : {adduct}")
    print(f"precursor_mz   : {precursor_mz}")
    assert abs(neutral_mass - 130.0) < 1e-9, (
    f"Expected 130.0, got {neutral_mass}"
    )
    print(f"calculated NM  : {neutral_mass}")
    print()

    print("=" * 78)
    print("ALL TESTS PASSED")
    print("=" * 78)
    print()

