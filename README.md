[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3810/) [![PyPI version](https://badge.fury.io/py/kudos.svg)](https://badge.fury.io/py/kudos)
# Kudos
## Overview
### What it's for
Public health case definitions often take the form of predictive checklists. The WHO, for example, defines [influenza-like illness](https://www.who.int/teams/global-influenza-programme/surveillance-and-monitoring/case-definitions-for-ili-and-sari) (ILI) as an acute respistoray infection with fever, cough, and an onset in the past 10 days; and the CDC defines a probable case of [pertussis (whooping cough)](https://ndc.services.cdc.gov/case-definitions/pertussis-2020/) as the presence of paroxysms of coughing, inspiratory whoop, post-coughing vomiting, or apnea for at least 2 weeks (or fewer than 2 weeks with exposure to a known case. Kudos is a Python package that helps you develop and test these kinds of case definitions using combinatorial optimization.

### Who it's for
Kudos was written with epidemiologists, biostatisticians, data scientists, and other data-savvy public health practitioners in mind. That being said, the code is subject-matter-agnostic, and so it can be used by anyone looking to build high-performance predictive checklists.

### How it works
Kudos use three kinds of combinatorial optimization methods to develop case definitions: linear programming (1); nonlinear programming; and brute-force search (2, 3). The first two methods are good for quickly finding a near-optimal definition based on your data, and the third method is good for exploring the full range of possible definitions. All of them figure out which combination of predictors (often symptoms) has the best classification performance relative to the reference standard you've specified (often a pathogen-specific like test like PCR or viral culture).

## Getting Started
### Installation
The easiest way to install Kudos is with pip: 

```sh
pip install kudos
```

The package is available on PyPI, so you can also use any standard package manager to fetch the code and handle the installation. If you'd like to contribute, fork the package first, and then install the dependencies manually.

```sh
git clone https://github.com/YOURNAME/kudos.git
cd kudos
pip install -r requirements.txt
```

### Software requirements
The package was written in Python 3.8, and because of some recent-ish changes to the `multiprocessing` package, it will not run on anything lower. It requires a few standard dependencies, like `numpy`, `scikit-learn`, and `seaborn`, but it will check for those during installation and add them if they're missing.

### Hardware
Kudos is best run on a scientific workstation or cloud instance with a decent amount of RAM and lots of processors. If you're using something less substantial, the optimizers will still work, but you may need to whittle down your dataset first if it has a large number of predictors. Regardless of hardware, the `FullEnumeration` (i.e., brute-force search) can take a long time to run, so keep that in mind when setting up the optimization.

## Using Kudos
### Interactive
Kudos is designed to be used interactively. Let's say you have a dataset named `data` with an outcome `y` and some number of predictors `X`. Finding a good case definition is as easy as fitting one of the models in the `optimizers` module.

```python
import pandas as pd
from optimizers import IntegerProgram

data = pd.read_csv('data.csv')
X = data[X_columns]
y = data[y_column]

ip = IntegerProgram()
ip.fit(X, y)
```

Once the solver finishes, it saves the optimal definition in the `results` attribute.

```python
ip.results
```

Seeing who meets the case definition in a new batch of data is just as easy.

```python
new_data = pd.read_csv('new_data.csv')
new_X = new_data[X_columns]
meets_definition = ip.predict(new_X)
```

The other optimizers, `FullEnumeration` and `NonlinearApproximation`, have the same functionality, and `FullEnumeration` also lets you do some visualizations with the candidate case definitions. For more info, see the [demo notebook](demo.ipynb).

### Command-line
Coming soon.

### Streamlit
Coming soon.

## Frequently Asked Questions
### Metrics
1. **How do the linear programs decide which case definition is the best?**
The `IntegerProgram` needs a linear objective function to run, meaning it's limited to metrics that are linear combinations of the predictors and the candidate case definitions. [Youden's J index](https://en.wikipedia.org/wiki/Youden%27s_J_statistic) (sensitivity + specificity - 1) meets that criterion, and it's a reasonable measure of overall classification performance, so that's what we use. Because it's a relaxed version of the integer program, the `NonlinearApproximation` uses this metric, as well. 

2. **What if I care more about sensitivity than specificity, or vice versa?**
You can change how much weight each component of the J index receives by altering the with the `alpha` (sensitivity) and `beta` (specificity) parameters of the linear program you `.fit()`. 

3. **What about the full enumeration?**
If you want to optimize a different metric than the J index, you can use the `FullEnumeration` instead of the LP-based optimizers. It will accept [F-score](https://en.wikipedia.org/wiki/F-score) or [Matthews correlation coefficient](https://en.wikipedia.org/wiki/Phi_coefficient) (MCC), in addition to J, as targets for sorting, pruning, and plotting. 

### Computing resources
1. **How can I make the brute-force search run faster?**
   * Whittle down your feature space. The `optimizers.FeaturePruner` is one way to do that, but standard variable-selection procedures (e.g., 
   forward or backward selection) will also work.
   * Try a lower value for `max_n`. The default is 5, which should work well in most cases.
   * Turn off `use_reverse`, if it's on. Using it doubles the size of the feature space.
   * Turn off `compound`, if it's on. Using it substantially increases the number of combinations to try.
2. **How can I make the brute-force search use less memory?**
   * Set `share_memory` to `True` when you initialize the `FullEnumeration` object. This keeps `multiprocessing` from passing copies of the
   dataset to every process in the `Pool`. 
   * Make sure `prune` is turned on. This limits the number of combinations saved at each step in the search.
   * Use a smaller number for `batch_keep_n`. This decides how many combos to save when `prune` is turned on.
   * Use fewer predictors. See the first answer to question #1 above.
3. **How can I make the linear program run faster?**
   * Try a lower value for `max_n`. The default is `None`, which will take the longest.
   * Try a different solver. [OR-Tools](https://developers.google.com/optimization/introduction/python), which is what Kudos uses on the backend, has a few options available.

## References
1. Zhang H, Morris Q, Ustun B, Ghassemi M. Learning optimal predictive checklists. _Advances in Neural Information Processing Systems_. 2021 Dec 6;34:1215-29.
2. Reses HE, Fajans M, Lee SH, Heilig CM, Chu VT, Thornburg NJ, Christensen K, Bhattacharyya S, Fry A, Hall AJ, Tate JE. Performance of existing and novel surveillance case definitions for COVID-19 in household contacts of PCR-confirmed COVID-19. _BMC public health_. 2021 Dec;21(1):1-5.
3. Lee S, Almendares O, Prince-Guerra JL, Heilig CM, Tate JE, Kirking HL. Performance of Existing and Novel Symptom-and Antigen Testing-Based COVID-19 Case Definitions in a Community Setting. _medRxiv_. 2022 Jan 1.

**General disclaimer** This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm).  Github is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software.

## Related documents

* [Open Practices](docs/open_practices.md)
* [Rules of Behavior](docs/rules_of_behavior.md)
* [Thanks and Acknowledgements](docs/thanks.md)
* [Disclaimer](docs/DISCLAIMER.md)
* [Contribution Notice](docs/CONTRIBUTING.md)
* [Code of Conduct](docs/code-of-conduct.md)

## Overview

Describe the purpose of your project. Add additional sections as necessary to help collaborators and potential collaborators understand and use your project.
  
## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
The repository utilizes code licensed under the terms of the Apache Software
License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md)
and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/privacy.html](http://www.cdc.gov/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page are subject to the [Presidential Records Act](http://www.archives.gov/about/laws/presidential-records.html)
and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices
Please refer to [CDC's Template Repository](https://github.com/CDCgov/template)
for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/master/CONTRIBUTING.md),
[public domain notices and disclaimers](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md),
and [code of conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
