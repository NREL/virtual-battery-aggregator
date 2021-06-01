# virtual-battery-aggregator
An aggregator for distributed batteries using OCHRE battery models for the FAST-DERMS project

## Installation

### Using `pip`

This repo can be installed using pip by running:
```
pip install git+https://github.com/NREL/virtual-battery-aggregator
```

or by adding this to an `environment.yml` file when installing with conda:
```
 - pip:
    - git+https://github.com/NREL/virtual-battery-aggregator
```

### Direct download

To install from the command line, download this repo and install using `setup.py`:
```
cd virtual-battery-aggregator
python setup.py install
```

## Usage

For an example script, see `bin/run_aggregator.py`. This script includes battery aggregator
functions for:
 - initialization
 - aggregation
 - disaggregation (dispatch)
 - updating model parameters

For an example with an OCHRE battery model, see `bin/run_with_ochre.py`.
Note that the `ochre` package is required to run this script.

## License
Distributed under the BSD-3 License. See LICENSE for more information.

## Contact
Michael Blonsky - Michael.Blonsky@nrel.gov
