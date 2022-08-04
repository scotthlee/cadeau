# Kudos
## Overview
### What's it for?
Public health case definitions often take the form of predictive checklists. The WHO, for example, defines [influenza-like illness](https://www.who.int/teams/global-influenza-programme/surveillance-and-monitoring/case-definitions-for-ili-and-sari) (ILI) as an acute respistoray infection with fever, cough, and an onset in the past 10 days; and the CDC defines a probable case of [pertussis (whooping cough)](https://ndc.services.cdc.gov/case-definitions/pertussis-2020/) as the presence of paroxysms of coughing, inspiratory whoop, post-coughing vomiting, or apnea for at least 2 weeks (or fewer than 2 weeks with exposure to a known case. Kudos is a Python package that lets you develop and test these kinds of case definitions using combinatorial optimization.

### Who's it for?
Kudos was written mostly with practicing epidmiologists in mind. That being said, the code is subject-matter-agnostic, and so it can be used by anyone looking to build high-performance predictive checklists.

### How does it work?
Kudos use three kinds of combinatorial optimization methods to develop case definitions: linear programming (Zhang et al. 2021); nonlinear programming; and brute-force search (Reses et al. 2021). The first two methods are good for quickly finding a near-optimal definition based on your data, and the third method is good for exploring the full range of possible definitions. In all cases, the solver will figure out which combination of predictors (often symptoms) has the best classification performance relative to the reference standard you've specified (often a pathogen-specific like test like PCR or viral culture).

## Getting Started
### Installation
The easiest way to install Kudos is with pip: `pip install kudos`. The package is available on PyPI, though, so you should be able to use any standard package manager to fetch the code and handle the installation. 

### Software requirements
The package was written in Python 3.8, but it should run fine with higher (but not lower) versions. The package requires a few standard dependencies, like `numpy`, `scikit-learn`, and `seaborn`, but it will check for those during installation and add them if they're missing.

### Hardware
Kudos is best run on a scientific workstation or cloud instance with a decent amount of RAM and lots of processors. If you're using something less substantial, the optimizers will still work, but you may need to use the `FeaturePruner` to whittle down your dataset  if it has a large number of predictors. Regardless of hardware, the `FullEnumeration` (i.e., brute-force search) can take a long time to run, so keep that in mind when setting up the optimization.

## Using Kudos
### Interactive
Kudos is designed to be used interactively, offering users a variety of tools for 

### Command-line
Coming soon.

### Streamlit
Coming soon.

## References
1. Zhang H, Morris Q, Ustun B, Ghassemi M. Learning optimal predictive checklists. _Advances in Neural Information Processing Systems_. 2021 Dec 6;34:1215-29.
2. Reses HE, Fajans M, Lee SH, Heilig CM, Chu VT, Thornburg NJ, Christensen K, Bhattacharyya S, Fry A, Hall AJ, Tate JE. Performance of existing and novel surveillance case definitions for COVID-19 in household contacts of PCR-confirmed COVID-19. _BMC public health_. 2021 Dec;21(1):1-5.

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
