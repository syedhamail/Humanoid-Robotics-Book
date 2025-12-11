# ROS 2 Humble/Iron Installation Best Practices on Ubuntu 22.04

This document outlines the best practices for installing ROS 2 Humble and Iron on Ubuntu 22.04, based on official documentation and community recommendations.

## General Best Practices (Applicable to both Humble and Iron)

1.  **System Update**: Always begin by updating and upgrading your system packages to ensure the latest versions and to prevent conflicts:
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```
2.  **UTF-8 Locale**: Verify and configure your system to use a UTF-8 locale. Incorrect locale settings can cause issues with ROS 2 tools.
    ```bash
    sudo apt install locales
    sudo locale-gen en_US.UTF-8
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LANG=en_US.UTF-8
    ```
3.  **Install from Debian Packages**: The recommended method is to install using `apt` via Debian packages.
4.  **Enable Ubuntu Universe Repository**: Ensure the Ubuntu Universe repository is enabled:
    ```bash
    sudo add-apt-repository universe
    ```
5.  **Add ROS 2 GPG Key and Repository**: Add the official ROS 2 GPG key and repository to your system's sources list.
6.  **Install Desktop Variant**: For a comprehensive installation, including development tools (RViz, demos, tutorials), install the `desktop` variant (e.g., `ros-humble-desktop`).
7.  **Source Setup Script**: After installation, source the ROS 2 setup script (`source /opt/ros/<ros_distro>/setup.bash`) in each new terminal or add it to your `~/.bashrc` for global availability.
8.  **Install Development Tools**: For creating and building ROS packages, install `ros-dev-tools`.

## Specific Best Practices for ROS 2 Humble

*   Follow the general best practices above.
*   The main installation package is `ros-humble-desktop`.

## Specific Best Practices for ROS 2 Iron

*   **Critical Pre-installation Update**: It is crucial to update `systemd` and `udev`-related packages *before* installing ROS 2 Iron on a freshly installed Ubuntu 22.04 system. Failure to do so can lead to critical system packages being removed.
*   The main installation package is `ros-iron-desktop`.

## Sources:
- [note.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGbTa3nWUAE7Y4PHb2TgIwq6k96WKCa1TIEdmqQQ5r6Mp8zvMBoLNP48MgqWIQVe6g4VgoH5AviWwaV3EQkG8ocFfGu6QxYoTO7VhZdUcCKWQ_89J8lnOovaevwqJKb8LHu-2Q-_tNnaNea)
- [upc.edu](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEnP0evOwj-fZhYXIm-3B2ahP5BdpmfMJTSa4gFKlfEN_8KzcS6206ph2qnfSSf9QTShRFnpmWZJ2JB8Xqbehq5C9pwo6yITBrH6J532Zadjbozl0iOzUceqCr8IbyLqnXM6lvG33m6Qk3V5rnEHTmR6xRwZgZhups9-UIJpakNOwj8-20GGlZcwJrRE_cV__c=)
- [foxglove.dev](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE-ynVjQ_-C-x67aGnjS6YRe1A7fk4qrqA06TbRjfBuDul_GchQp1AoeumIkGoSE-gFBhldXx0tlc3iKmDJCKNYcPvSj1GEo_zBmfLc_S4e_ZOWlAV7eOJqoML5Mb-60EWsKV6xi6GM4NpTB9zjkUNh8fA_pas=)
- [automaticaddison.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE61bxT0e_rEPfo4MQpmsYoJm1__uIGt0WkEjO0UEyNjOG7HqwRZD5iZO3mhZucyFcBsQpod9qsYO9Duviwh4DIHtymtxW0adaEPg9_o24axB8UTkb_urzhbo959zMIij3tuUklOFTLQ2O9RzMYZgMqhVxFnl27FjWeuE09Qu4DIOUw)
- [ros.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQENcIY0pHSq4MS3EXjnY8xUsV-DL4N9708k1UmS3wVu1SkZTU55b0mA3Xo1ajEfiyrpfEVnyXHS1wJsaMkKh_Nd3sJ0UvNQIqh-ANCCln8bwhT6OUeLOiybvA8_F8HcLpaY8Ya1DGnLFeMvvMpOaa8TgfEfPZxAk2BRudKY9Q==)
- [foxglove.dev](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHeEMdMc6d7WUwk-T-cwN_usfb-RrCY-6B2mL93MEe34LcSCST7ZMSRx5bxEByX4F_uL4BMGT6Wh3HHxaO24_AXCsxj5VGp3kjcOWOHedDikmYgrCYNNpaGyE5DbbWpnzQreBz-QAx_6X5baSjwpmLmHFZE)
- [theconstruct.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH--H7zqd4Dw7vfvjCNu_7d2eKe4hSXhUbbSjE0wwQkmmVOlZ1DZZuHCie_ZoCV6stMKJCLG47LaDp9CYokGDmaUa0HJgdjVvQHOUADFAHWPW4kyVj7H0WBlFg1kjIowGadGr8Z_qfaGQ3Wqd4BoUsANrzLSTB7ARWVs5x-MtTQbADnmjw=)
- [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHj-ASdsT6gBdO8AIyodCUCIlDaCC0SPf6LUieYzzHx90Tg35njonFa5Ml_HX9I_F8xv0hVFSvPdxbMrE3OqA2MC7xelP6bHtsHhcnpHESCAExQXCm4FKNqGgss4CUX4oef3BICQ1g=)
- [reddit.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFsf3HlnwznE7AGSL8t5yDt_BvggkL3U92gi95uLuWCNlsuE3R6C645XUejNBmUuzKfJZMkrPceWq6AN4ZwGGLMNOrNiy2KZgkGnnYq1sjDWNbJLoOiR2KYPrv35CQe_K0BucSOmeksuRYqStWdf6S4sCmJoZbk0tXcFv4-uLv2gdRWilsQrqy_-dIsSHMHjiniCqPXat7p_yv3cSo_)
- [aleksandarhaber.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG5BgsM1Jf8_iDhNyVv4HHiNvcVNmQLUfeMWvThFTk5P1fhmXB20p-LKfdqSBdG6rbqZQu_NA3BDM1MC8mSk5MT5iUSjUuXmbo8yCExnUeSMyG5wMroEhK0L0_CavLDdYCZB1KzQwGGURtsMBWiucMrTxKkwY4BpYIheer4sT2kyiENHwe6REzuCDCuq2SZ-pLzI_oCrwEISOAcY9w=)
