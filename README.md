# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/cryocast/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                               |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| cryocast/\_\_init\_\_.py                                           |        9 |        0 |    100% |           |
| cryocast/callbacks/\_\_init\_\_.py                                 |        7 |        0 |    100% |           |
| cryocast/callbacks/activation\_saver.py                            |       92 |        0 |    100% |           |
| cryocast/callbacks/ema\_weight\_averaging\_callback.py             |       21 |        0 |    100% |           |
| cryocast/callbacks/media\_logging\_callback.py                     |      151 |        0 |    100% |           |
| cryocast/callbacks/metric\_summary\_callback.py                    |      106 |        1 |     99% |        93 |
| cryocast/callbacks/prediction\_writer.py                           |      203 |       19 |     91% |71-74, 79-80, 164-168, 282-283, 340-344, 346-350, 369, 392-393, 409-413 |
| cryocast/callbacks/unconditional\_checkpoint.py                    |       26 |        0 |    100% |           |
| cryocast/cli/\_\_init\_\_.py                                       |        2 |        0 |    100% |           |
| cryocast/cli/datasets.py                                           |       53 |        1 |     98% |       129 |
| cryocast/cli/evaluate.py                                           |       26 |        3 |     88% | 37-42, 93 |
| cryocast/cli/feature\_importance.py                                |       15 |        1 |     93% |        26 |
| cryocast/cli/hydra.py                                              |       29 |        0 |    100% |           |
| cryocast/cli/main.py                                               |       28 |        1 |     96% |        61 |
| cryocast/cli/sweep.py                                              |       90 |        1 |     99% |       216 |
| cryocast/cli/train.py                                              |       16 |        1 |     94% |        55 |
| cryocast/compatibility/\_\_init\_\_.py                             |       17 |        0 |    100% |           |
| cryocast/compatibility/lightning/\_\_init\_\_.py                   |        9 |        0 |    100% |           |
| cryocast/compatibility/lightning/xpu\_accelerator.py               |       72 |       39 |     46% |35-42, 55, 60-62, 77-111, 125-127, 133, 144-147, 152 |
| cryocast/compatibility/torch/\_\_init\_\_.py                       |        4 |        0 |    100% |           |
| cryocast/compatibility/torch/patch\_interpolate\_antialias.py      |       11 |        8 |     27% |     14-26 |
| cryocast/compatibility/torch/patch\_open\_file\_limit.py           |       18 |        2 |     89% |     34-35 |
| cryocast/compatibility/torch/patch\_parameter\_deepcopy.py         |       12 |        7 |     42% |     16-28 |
| cryocast/config/\_\_init\_\_.py                                    |        0 |        0 |    100% |           |
| cryocast/data/\_\_init\_\_.py                                      |        5 |        0 |    100% |           |
| cryocast/data/calendar\_day\_climatology.py                        |       40 |        0 |    100% |           |
| cryocast/data/combined\_dataset.py                                 |       56 |        0 |    100% |           |
| cryocast/data/common\_data\_module.py                              |      148 |        5 |     97% |229-230, 248-254 |
| cryocast/data/single\_dataset.py                                   |      148 |        2 |     99% |   313-318 |
| cryocast/data/variable\_selection.py                               |       50 |        0 |    100% |           |
| cryocast/exceptions.py                                             |        3 |        0 |    100% |           |
| cryocast/feature\_importance.py                                    |       26 |        0 |    100% |           |
| cryocast/geotools/\_\_init\_\_.py                                  |       11 |        0 |    100% |           |
| cryocast/geotools/geographic\_field.py                             |       37 |        0 |    100% |           |
| cryocast/geotools/geographic\_grid.py                              |       76 |        0 |    100% |           |
| cryocast/geotools/geographic\_metadata.py                          |       88 |        0 |    100% |           |
| cryocast/geotools/grid\_factory.py                                 |       44 |        0 |    100% |           |
| cryocast/geotools/regrid\_forecast.py                              |       74 |        0 |    100% |           |
| cryocast/geotools/reproject.py                                     |       26 |        0 |    100% |           |
| cryocast/ingestion/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| cryocast/ingestion/data\_downloader.py                             |      144 |       52 |     64% |67-68, 105-145, 149-166, 195, 200-203, 212-214, 223-224, 238-248 |
| cryocast/ingestion/downloaders.py                                  |       11 |        0 |    100% |           |
| cryocast/ingestion/filters/\_\_init\_\_.py                         |       13 |        0 |    100% |           |
| cryocast/ingestion/filters/nan\_to\_num\_filter.py                 |        9 |        0 |    100% |           |
| cryocast/ingestion/filters/reproject\_filter.py                    |       31 |        0 |    100% |           |
| cryocast/ingestion/filters/set\_geography\_filter.py               |       19 |        0 |    100% |           |
| cryocast/ingestion/postprocessors/\_\_init\_\_.py                  |        4 |        0 |    100% |           |
| cryocast/ingestion/postprocessors/composite.py                     |        9 |        0 |    100% |           |
| cryocast/ingestion/postprocessors/ipostprocessor.py                |        8 |        0 |    100% |           |
| cryocast/ingestion/postprocessors/status\_flag\_mask\_generator.py |       44 |        5 |     89% |39-40, 50-51, 64 |
| cryocast/ingestion/postprocessors/synthetic\_mask\_generator.py    |       22 |        2 |     91% |     24-25 |
| cryocast/ingestion/preprocessors/\_\_init\_\_.py                   |        2 |        0 |    100% |           |
| cryocast/ingestion/preprocessors/composite.py                      |        9 |        0 |    100% |           |
| cryocast/ingestion/preprocessors/ipreprocessor.py                  |        8 |        0 |    100% |           |
| cryocast/ingestion/sources/\_\_init\_\_.py                         |       24 |        0 |    100% |           |
| cryocast/ingestion/sources/argo.py                                 |       89 |       12 |     87% |60-61, 105-106, 137-142, 234-240 |
| cryocast/ingestion/sources/ftp.py                                  |       43 |        0 |    100% |           |
| cryocast/ingestion/sources/lazy\_argopy.py                         |       13 |        2 |     85% |     29-30 |
| cryocast/ingestion/sources/synthetic.py                            |       27 |       10 |     63% |33-42, 50-88 |
| cryocast/loggers/\_\_init\_\_.py                                   |        2 |        0 |    100% |           |
| cryocast/loggers/local\_file\_logger.py                            |       51 |        0 |    100% |           |
| cryocast/losses/\_\_init\_\_.py                                    |        5 |        0 |    100% |           |
| cryocast/losses/amse\_loss.py                                      |      152 |       17 |     89% |281-285, 288-289, 294-315 |
| cryocast/losses/build\_loss.py                                     |       16 |        0 |    100% |           |
| cryocast/losses/lead\_time\_weighted\_loss.py                      |       44 |        0 |    100% |           |
| cryocast/losses/rmse\_loss.py                                      |        9 |        0 |    100% |           |
| cryocast/metrics/\_\_init\_\_.py                                   |       12 |        0 |    100% |           |
| cryocast/metrics/base\_daily\_metric.py                            |       39 |        4 |     90% |54-58, 65, 78 |
| cryocast/metrics/base\_ice\_area\_metric.py                        |       32 |        3 |     91% |37, 73, 92 |
| cryocast/metrics/centroid\_error.py                                |       31 |        2 |     94% |     53-56 |
| cryocast/metrics/distance\_averaged\_iee.py                        |       31 |        1 |     97% |       102 |
| cryocast/metrics/fss.py                                            |       70 |        3 |     96% |132-135, 184 |
| cryocast/metrics/helpers.py                                        |       30 |        2 |     93% |    88, 98 |
| cryocast/metrics/icenet\_accuracy.py                               |       24 |        1 |     96% |        56 |
| cryocast/metrics/iiee.py                                           |        9 |        0 |    100% |           |
| cryocast/metrics/mae.py                                            |        5 |        0 |    100% |           |
| cryocast/metrics/rmse.py                                           |        7 |        0 |    100% |           |
| cryocast/metrics/sie.py                                            |       14 |        2 |     86% |     21-22 |
| cryocast/metrics/spatial\_mean\_trace.py                           |        8 |        0 |    100% |           |
| cryocast/metrics/ssim.py                                           |       43 |        2 |     95% |     55-56 |
| cryocast/model\_service.py                                         |      287 |        4 |     99% |57-58, 213-214 |
| cryocast/models/\_\_init\_\_.py                                    |        6 |        0 |    100% |           |
| cryocast/models/base\_model.py                                     |      130 |        3 |     98% |166, 170, 174 |
| cryocast/models/climatology.py                                     |       15 |        0 |    100% |           |
| cryocast/models/common/\_\_init\_\_.py                             |       24 |        0 |    100% |           |
| cryocast/models/common/activations.py                              |        2 |        0 |    100% |           |
| cryocast/models/common/channel\_adaptor.py                         |       18 |        0 |    100% |           |
| cryocast/models/common/conv\_block\_common.py                      |        8 |        0 |    100% |           |
| cryocast/models/common/conv\_block\_downsample.py                  |       13 |        2 |     85% |     45-46 |
| cryocast/models/common/conv\_block\_upsample.py                    |       19 |        4 |     79% |55-56, 59-60 |
| cryocast/models/common/conv\_lstm\_cell.py                         |       26 |        0 |    100% |           |
| cryocast/models/common/conv\_norm\_act.py                          |        9 |        0 |    100% |           |
| cryocast/models/common/conv\_norm\_act\_upsample.py                |       10 |        0 |    100% |           |
| cryocast/models/common/freezable.py                                |        7 |        0 |    100% |           |
| cryocast/models/common/gated\_attention.py                         |       52 |       27 |     48% |23-46, 49-50, 92-116, 124-130, 135-136, 139-141 |
| cryocast/models/common/glumb\_conv.py                              |       23 |        1 |     96% |        71 |
| cryocast/models/common/lite\_mla.py                                |       30 |        2 |     93% |     44-45 |
| cryocast/models/common/mask.py                                     |       24 |        2 |     92% |     52-56 |
| cryocast/models/common/normalisations.py                           |       20 |        2 |     90% |     13-14 |
| cryocast/models/common/normalised\_fold.py                         |       19 |        0 |    100% |           |
| cryocast/models/common/patchembed.py                               |       13 |        0 |    100% |           |
| cryocast/models/common/permute.py                                  |        7 |        0 |    100% |           |
| cryocast/models/common/res\_block.py                               |       16 |        0 |    100% |           |
| cryocast/models/common/residual\_downsample.py                     |       20 |        5 |     75% | 54-59, 76 |
| cryocast/models/common/residual\_upsample.py                       |       15 |        1 |     93% |        67 |
| cryocast/models/common/resizing\_interpolation.py                  |       13 |        0 |    100% |           |
| cryocast/models/common/restrict\_range.py                          |       14 |        0 |    100% |           |
| cryocast/models/common/shift.py                                    |       14 |        8 |     43% |10-14, 20-24 |
| cryocast/models/common/skip\_connection.py                         |       22 |        2 |     91% |     66-67 |
| cryocast/models/common/time\_embed.py                              |        8 |        0 |    100% |           |
| cryocast/models/common/transformerblock.py                         |       12 |        0 |    100% |           |
| cryocast/models/common/weighted\_upsample.py                       |       16 |        0 |    100% |           |
| cryocast/models/ddpm.py                                            |      140 |        5 |     96% |363-371, 607 |
| cryocast/models/decoders/\_\_init\_\_.py                           |        6 |        0 |    100% |           |
| cryocast/models/decoders/base\_decoder.py                          |       36 |        2 |     94% |   123-124 |
| cryocast/models/decoders/cnn\_decoder.py                           |       44 |        4 |     91% |91-92, 152-153 |
| cryocast/models/decoders/deep\_compression\_decoder.py             |       42 |        6 |     86% |60-61, 63-64, 66-67 |
| cryocast/models/decoders/naive\_linear\_decoder.py                 |       15 |        0 |    100% |           |
| cryocast/models/decoders/piecewise\_decoder.py                     |       27 |        0 |    100% |           |
| cryocast/models/diffusion/\_\_init\_\_.py                          |        3 |        0 |    100% |           |
| cryocast/models/diffusion/gaussian\_diffusion.py                   |       54 |        4 |     93% |42, 46-50, 200 |
| cryocast/models/diffusion/unet\_diffusion.py                       |       78 |        1 |     99% |       269 |
| cryocast/models/encode\_process\_decode.py                         |      149 |       19 |     87% |63-68, 92-97, 109-113, 139-145, 197-202, 368-373, 416-422, 485-489 |
| cryocast/models/encoders/\_\_init\_\_.py                           |        7 |        0 |    100% |           |
| cryocast/models/encoders/base\_encoder.py                          |       36 |        2 |     94% |     59-60 |
| cryocast/models/encoders/cnn\_encoder.py                           |       25 |        0 |    100% |           |
| cryocast/models/encoders/deep\_compression\_encoder.py             |       42 |        6 |     86% |61-62, 64-65, 67-68 |
| cryocast/models/encoders/naive\_linear\_encoder.py                 |       15 |        0 |    100% |           |
| cryocast/models/encoders/piecewise\_encoder.py                     |       20 |        0 |    100% |           |
| cryocast/models/encoders/reprojection\_encoder.py                  |       36 |        0 |    100% |           |
| cryocast/models/multistage/\_\_init\_\_.py                         |        4 |        0 |    100% |           |
| cryocast/models/multistage/decoder\_stage.py                       |       51 |        0 |    100% |           |
| cryocast/models/multistage/encoder\_stage.py                       |       28 |        0 |    100% |           |
| cryocast/models/multistage/processor\_stage.py                     |       26 |        0 |    100% |           |
| cryocast/models/persistence.py                                     |       17 |        0 |    100% |           |
| cryocast/models/processors/\_\_init\_\_.py                         |        8 |        0 |    100% |           |
| cryocast/models/processors/base\_processor.py                      |       30 |        2 |     93% |     39-43 |
| cryocast/models/processors/conv\_lstm.py                           |       52 |        0 |    100% |           |
| cryocast/models/processors/diffusion.py                            |      143 |        0 |    100% |           |
| cryocast/models/processors/gsta.py                                 |       22 |       12 |     45% |65-73, 99-110 |
| cryocast/models/processors/null.py                                 |       10 |        0 |    100% |           |
| cryocast/models/processors/unet.py                                 |       53 |        0 |    100% |           |
| cryocast/models/processors/vit.py                                  |       43 |        4 |     91% |41-42, 101-105 |
| cryocast/sweep/\_\_init\_\_.py                                     |        2 |        0 |    100% |           |
| cryocast/sweep/optuna\_sweep.py                                    |      125 |        2 |     98% |   239-240 |
| cryocast/sweep/parameters.py                                       |       98 |        1 |     99% |        76 |
| cryocast/sweep/sampler\_store.py                                   |       40 |        0 |    100% |           |
| cryocast/synthetic/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| cryocast/synthetic/shapes.py                                       |       81 |        0 |    100% |           |
| cryocast/synthetic/trajectories.py                                 |       77 |        9 |     88% |180-191, 207-211 |
| cryocast/types/\_\_init\_\_.py                                     |        7 |        0 |    100% |           |
| cryocast/types/annotations.py                                      |       12 |        0 |    100% |           |
| cryocast/types/complex\_datatypes.py                               |      132 |        1 |     99% |        37 |
| cryocast/types/constants.py                                        |        6 |        0 |    100% |           |
| cryocast/types/enums.py                                            |       28 |        0 |    100% |           |
| cryocast/types/protocols.py                                        |       17 |        0 |    100% |           |
| cryocast/types/simple\_datatypes.py                                |       25 |        0 |    100% |           |
| cryocast/utils.py                                                  |       53 |        0 |    100% |           |
| cryocast/visualisations/\_\_init\_\_.py                            |        7 |        0 |    100% |           |
| cryocast/visualisations/dataset\_media\_writer.py                  |       46 |        0 |    100% |           |
| cryocast/visualisations/difference\_panel.py                       |       58 |        0 |    100% |           |
| cryocast/visualisations/land\_mask.py                              |       23 |        0 |    100% |           |
| cryocast/visualisations/matplotlib\_renderer.py                    |      137 |        1 |     99% |       286 |
| cryocast/visualisations/media\_annotator.py                        |       50 |        0 |    100% |           |
| cryocast/visualisations/media\_publisher.py                        |      114 |        1 |     99% |       184 |
| cryocast/visualisations/panel\_renderer.py                         |       87 |        2 |     98% |     60-61 |
| cryocast/visualisations/style\_resolver.py                         |       30 |        0 |    100% |           |
| **TOTAL**                                                          | **6445** |  **358** | **94%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/alan-turing-institute/cryocast/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/cryocast/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/alan-turing-institute/cryocast/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/cryocast/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Falan-turing-institute%2Fcryocast%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/cryocast/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.