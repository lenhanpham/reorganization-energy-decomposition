### Main purpose of this project is to decompose reorganization energy into contribution of vibrational modes using Duschinsky matrix, and then calculate Huang-Rhys factors of each mode. 

### The project in under implementation and not ready for use.



reorganization-energy-decomposition/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── matrix_operations.py    # Replace ddiag.for, dmpower.for, dmatinv.for
│   │   ├── coordinates.py          # Handle coordinate transformations
│   │   └── force_constants.py      # Handle force constant calculations
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── gaussian_parser.py      # Handle Gaussian file parsing
│   │   └── constants.py            # Physical constants and parameters
│   └── programs/
│       ├── __init__.py
│       ├── dushin.py              # Replace dushin.for
│       ├── displace.py            # Replace displace.for
│       └── compare_geom.py        # Replace compare-geom.for
├── tests/
│   └── __init__.py
├── requirements.txt
└── setup.py