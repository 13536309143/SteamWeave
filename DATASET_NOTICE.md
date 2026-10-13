# Dataset provenance and audit

The original dataset is included at the repository owner's request in `Multimodal data for prognostics/`. Original files are retained, including the upstream `Readme.docx` and its attribution request.

Source: [LASAR Public Benchmark](https://www.lasar.polimi.it/?page_id=2970).

Dataset-associated paper: Yang, Z., Baraldi, P., and Zio, E. (2021). *A multi-branch deep neural network model for failure prognostics based on multimodal data.* Journal of Manufacturing Systems, 59, 42–50. [DOI](https://doi.org/10.1016/j.jmsy.2021.01.007).

The data remains an upstream research artifact. Inclusion in this repository does not assign a new license to it. Consult the original authors/source for reuse conditions beyond the attribution provided here.

This local copy contains 50 device directories and 7,626 records. Devices 1–20 have identical numerical, observation-time and text histories to devices 31–50, respectively. Grouping by these input histories yields 30 distinct trajectory groups. Seventeen pairs also have byte-identical entire records, targets and image files. This is an audit of the supplied copy; it does not establish which dataset version or trajectory realization was used in the original paper.

Evaluation retains all records, keeps each duplicate group in one partition and balances target inspection events during training. The device-ID-only five-fold protocol is explicitly diagnostic because copies can cross partitions.

数据集按仓库所有者要求收录，保留原始说明与作者引用。对本地副本的重复轨迹审计不等同于对原论文数据版本的判断。数据集的既有权利与使用条件不因收录而改变。
