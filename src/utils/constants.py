from dataclasses import dataclass

@dataclass
class Constants:
    """Physical constants and conversion factors"""
    AU_TO_ANGSTROM: float = 0.529177249
    AU_TO_CM: float = 219474.63067
    AMU_TO_AU: float = 1822.888486
    EV_TO_CM: float = 8065.817
    
    # Maximum dimensions (from original Fortran parameters)
    MAX_ATOMS: int = 300
    MAX_COORDS: int = 900  # 3 * MAX_ATOMS
    MAX_INTERNAL: int = 3600  # 12 * MAX_ATOMS

    # Atomic symbols (replacing atsym common block)
    ATOMIC_SYMBOLS: dict = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 10: 'Ne'  # Add more elements as needed
    }