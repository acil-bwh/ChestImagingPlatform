ChestImagingPlatform
====================

The Brigham and Women's Hospital (BWH) Applied Chest Imaging Lab's (ACIL) Chest Imaging Platform private repository

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