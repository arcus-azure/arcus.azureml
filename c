[33mcommit 161ae37c8fa61fdf4d7aa83f240d5f088c71298b[m[33m ([m[1;36mHEAD -> [m[1;32mmaster[m[33m)[m
Author: Sriram Narayanan(Codit) <sriram.narayanan@codit.eu>
Date:   Thu Apr 30 12:35:35 2020 +0200

    Applied NamingConventions and added optional root_folder parameter

[33mcommit 6e69456868f658fd02337072d533b081c22f59bd[m
Author: Sriram Narayanan(Codit) <sriram.narayanan@codit.eu>
Date:   Sat Apr 25 18:45:05 2020 +0200

    Added project utils

[33mcommit cf8b71ce9bb427589b31f7bd4d1bbaf7d0ba1ac1[m
Author: Sriram Narayanan(Codit) <sriram.narayanan@codit.eu>
Date:   Fri Apr 24 23:19:16 2020 +0200

    added ProjectUtils

[33mcommit 74e5c9d128a9dde8978bed3ebfca9eab08661d09[m[33m ([m[1;31morigin/master[m[33m)[m
Author: Sam Vanhoutte <SamVanhoutte@users.noreply.github.com>
Date:   Fri Apr 24 16:13:21 2020 +0200

    Update cd-build.yml for Azure Pipelines

[33mcommit ee56bf99cf0b90d12efe967e1a9a85a5f833dba4[m
Author: Sam Vanhoutte <SamVanhoutte@users.noreply.github.com>
Date:   Fri Apr 24 16:11:19 2020 +0200

    Update cd-build.yml for Azure Pipelines

[33mcommit e90aec48046e87be97733a6db9b3ccdac4db4164[m
Author: Sam Vanhoutte <SamVanhoutte@users.noreply.github.com>
Date:   Fri Apr 24 15:18:56 2020 +0200

    changed directory structure to comply with package requirements (#39)
    
    Co-authored-by: Sam <sam.vanhoutte@marchitec.be>

[33mcommit d7840264a3dc164a4d0776f5325685d76c819c54[m
Author: Sam Vanhoutte <SamVanhoutte@users.noreply.github.com>
Date:   Mon Apr 20 14:00:04 2020 +0200

    Update docs (#28)
    
    * change package name from azure-ml to azureml
    
    * update gitignor
    
    Co-authored-by: Sam <sam.vanhoutte@marchitec.be>

[33mcommit 0116bfef33ca27f0b6c6646fc6009db9b336de28[m
Author: Sam Vanhoutte <SamVanhoutte@users.noreply.github.com>
Date:   Mon Apr 20 12:05:44 2020 +0200

    Update cd-build.yml for Azure Pipelines

[33mcommit f7072b19fb5a17a4614de063e35a889628a23305[m
Author: Tom Kerkhove <kerkhove.tom@gmail.com>
Date:   Mon Apr 20 11:53:04 2020 +0200

    Remove local template to determine build version (#27)

[33mcommit 00a1bace7326d2c90f0cdb37b1867c4795508948[m
Author: Sam Vanhoutte <SamVanhoutte@users.noreply.github.com>
Date:   Mon Apr 20 11:49:24 2020 +0200

    Forcing PR to trigger master merge build
    
    * update master build (IndividualCI)
    
    * Update ci-build.yml
    
    * Update ci-build.yml
    
    * Update ci-build.yml
    
    * use devops template now
    
    devops template PR was approved
    
    * removed version yml

[33mcommit 679dcce66c3fcec0c7fc85c0510d94cd2362515e[m
Author: Tom Kerkhove <kerkhove.tom@gmail.com>
Date:   Mon Apr 20 10:43:19 2020 +0200

    Rough approach for build version determination (#25)
    
    * Rough approach for build version determination
    
    * create python pr version number
    
    * Update ci-build.yml
    
    * pad pr number with zeros and remove hyphen
    
    * Check if underscore works
    
    Co-Authored-By: Tom Kerkhove <kerkhove.tom@gmail.com>
    
    * Update determine-build-version.yml
    
    * Update determine-build-version.yml
    
    * Update determine-build-version.yml
    
    * This is the only working option, it seems
    
    * Update determine-build-version.yml
    
    * Set version to 1.0.0
    
    * leverage templates
    
    Co-authored-by: Sam Vanhoutte <SamVanhoutte@users.noreply.github.com>

[33mcommit ad49a6c0b0a5b3ac1bccd1d06cc8ce138dda9ec5[m
Author: Sam Vanhoutte <SamVanhoutte@users.noreply.github.com>
Date:   Mon Apr 20 08:38:20 2020 +0200

    Build increment version (#24)
    
    * Test CI increment (fixed value)
    
    * Use build number as version
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines
    
    * remove version tests
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines
    
    * hierarchy
    
    * Update ci-build.yml for Azure Pipelines
    
    * Update ci-build.yml for Azure Pipelines

[33mcommit e8986e373bc2a0ba6346a9d7c4cd0b382efcd223[m
Author: Sam Vanhoutte <SamVanhoutte@users.noreply.github.com>
Date:   Mon Apr 20 08:32:04 2020 +0200

    Create release build (#23)
    
    * add initial build
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    Co-authored-by: Sam <sam.vanhoutte@marchitec.be>

[33mcommit b4f51e7d3c1591ff2daac322c2956fdc1173e27f[m
Author: Sam Vanhoutte <SamVanhoutte@users.noreply.github.com>
Date:   Tue Apr 7 09:52:47 2020 +0200

    Initial package structure (#17)
    
    * add initial package structure
    
    * add requirements.txt
    
    * Set up CI with Azure Pipelines (#16)
    
    [skip ci]
    
    * added pytests
    
    * update build def
    
    * Update azure-pipelines.yml for Azure Pipelines
    
    * Update ci-build.yml
    
    * Delete azure-pipelines.yml
    
    file has been created/updated in the /build folder
    
    * Update README.md
    
    * Update README.md
    
    * Update ci-build.yml for Azure Pipelines
    
    * add version number
    
    * add release build
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * update version
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update cd-build.yml for Azure Pipelines
    
    * Update build/cd-build.yml
    
    Co-Authored-By: Tom Kerkhove <kerkhove.tom@gmail.com>
    
    * applying suggestions of PR review #17
    
    * update tests
    
    * updated package description path
    
    * Delete cd-build.yml
    
    Co-authored-by: Sam <sam.vanhoutte@marchitec.be>
    Co-authored-by: Tom Kerkhove <kerkhove.tom@gmail.com>

[33mcommit f9ebeaf548a7eadc88d43cb1b32d613ace8a1372[m
Author: stijnmoreels <9039753+stijnmoreels@users.noreply.github.com>
Date:   Wed Mar 25 12:25:06 2020 +0100

    Chore - move issue templates to '/.github' (#14)

[33mcommit f3cf03558a80d01513e303ab812f56afd6fad665[m
Author: Tom Kerkhove <kerkhove.tom@gmail.com>
Date:   Fri Mar 13 15:41:26 2020 +0100

    Fix failing Netlify builds (#13)

[33mcommit 737cd09cd8dc80ff9a88d242dcbba225dacbbd91[m
Author: Tom Kerkhove <kerkhove.tom@gmail.com>
Date:   Fri Mar 13 15:23:25 2020 +0100

    Update netlify.toml

[33mcommit 4423de7297a00e1db145b0f88b74a4a5643af5ee[m
Author: Tom Kerkhove <kerkhove.tom@gmail.com>
Date:   Fri Mar 13 15:19:41 2020 +0100

    Fix landing page

[33mcommit 06868b1b93a8f40e3a26ff8a35df89fcfa86350d[m
Author: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>
Date:   Fri Mar 13 15:06:50 2020 +0100

    Configure Renovate (#1)
    
    * Add renovate.json
    
    * Update renovate.json
    
    Co-authored-by: Renovate Bot <bot@renovateapp.com>
    Co-authored-by: Tom Kerkhove <kerkhove.tom@gmail.com>

[33mcommit 95a06c8fd3e9c758ce0b0c3669293fe1ad3cd0f5[m
Author: Tom Kerkhove <kerkhove.tom@gmail.com>
Date:   Fri Mar 13 15:06:01 2020 +0100

    Remove myself as code owner

[33mcommit 00863586b0ee87809ef28df3ff644d6dff5f8742[m
Merge: 1fd91ce 15d198a
Author: Tom Kerkhove <kerkhove.tom@gmail.com>
Date:   Fri Mar 13 15:05:05 2020 +0100

    Merge pull request #10 from tomkerkhove/remove-csharp-template
    
    Remove .NET template

[33mcommit 1fd91ce09c64ad865546d046b922e11136246b5e[m
Author: Tom Kerkhove <kerkhove.tom@gmail.com>
Date:   Fri Mar 13 08:52:03 2020 +0100

    Update CNAME to machine-learning.arcus-azure.net

[33mcommit fdbf3dbfa2a8d77656d42cbf1ef708e86bd3ecdd[m
Author: Tom Kerkhove <kerkhove.tom@gmail.com>
Date:   Thu Mar 12 17:06:48 2020 +0100

    Fix doc config

[33mcommit cbe3a21bc95004adbfdf77ebbcecd03861d177d3[m
Author: Tom Kerkhove <kerkhove.tom@gmail.com>
Date:   Thu Mar 12 16:48:48 2020 +0100

    Netlify configuration

[33mcommit 15d198ad0bc05002a6ebf4157d24fcf1003d674c[m
Author: Tom Kerkhove <kerkhove.tom@gmail.com>
Date:   Thu Mar 12 16:42:03 2020 +0100

    Remove .NET template

[33mcommit 631ac135355a1878e936fcc2c220b77334d7a76a[m
Author: Sam Vanhoutte <SamVanhoutte@users.noreply.github.com>
Date:   Thu Mar 12 13:11:00 2020 +0100

    Initial commit
