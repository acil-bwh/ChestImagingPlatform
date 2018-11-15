ChestImagingPlatform
====================

The Chest Imaging Platform (CIP) is an open-source package for image-based analysis and phenotyping of chest CT. The toolbox is developed and maintained by the Applied Chest Imaging Laboratory [ACIL](https://acil.med.harvard.edu) at Brigham and Women's Hospital, Harvard Medical School.

CIP is written in both C++ and python and provides both core classes as well as com- mand line tools that form the basis of the computational components. The library design is such that allows for extensibility so other groups and developers can contribute to the project

References
===============

1.	San José Estépar R, Ross JC, Harmouche R, Onieva J, Diaz AA, Washko GR. Chest Imaging Platform: An Open-Source Library and Workstation for Quantitative Chest Imaging. American Thoracic Society International Conference Abstracts American Thoracic Society; 2015. pp. A4975–A4975.doi:10.1164/ajrccm-conference.2015.191.1_MeetingAbstracts.A4975.

2.	Onieva J, Ross J, Harmouche R, Yarmarkovich A, Lee J, Diaz A, Washko GR, San José Estépar R. Chest Imaging Platform: an open-source library and workstation for quantitative chest imaging. Int J Comput Assist Radiol Surg 2016;11 Suppl 1:S40–S41.


Building Instructions
============
You can find instructions about building CIP [here] [3].
The building process will resovle dependencies for CIP.

Information for Users
==========================

We are happy to receive questions, comments, bug reports, etc. Please post any questions to _chestimagingplatform-users@googlegroups.com_

Information for Developers
==========================

Commit Codes
------------
The following codes should prepend comments to commits

* ENH: Source code enhancment
* DEB: A bug fix
* COMP: A change needed to get the source to compile/link
* STYLE: Style change
* ADD: Addition of a new file or directory to the repository

Coding Conventions
------------------

For C++ code, we follow [ITK Coding Style Guide] [1]. For python code, we follow the [PEP 8 - Style Guide for Python Code] [2]. Highlights and additional ChestImagingPlatform-specifc guidelines for C++ code are as follows

* Avoid white space immediately inside brackets, braces and parentheses
* Avoid white space immediately before a comma, semicolon, or colon
* Don't use spaces around the = sign when used to indicate a keyword argument or a default parameter value
* Use '//' for each line of a comment block, followed by a space, followed by a capitalized letter if beginning a sentence or phrase. This pertains to comments not meant to appear in the documentation.
* Follow Doxygen style comments as per recommendations in the [ITK Coding Style Guide] [1] where appropriate
* Use one space immediately before and after: +, -, <<, >>, &&, ||, +=, *=, -=, \=, ==, <, >, =
* No white space should be used for operators * and /

[1]: http://www.vtk.org/Wiki/ITK/Coding_Style_Guide  "ITK Coding Style Guide"
[2]: http://www.python.org/dev/peps/pep-0008/        "PEP 8"
[3]: https://github.com/acil-bwh/ChestImagingPlatform/wiki#building-instructions

Acknowledgments
=================
CIP is funded by the National Heart, Lung, And Blood Institute of the National Institutes of Health under Award Number R01HL116931. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.
