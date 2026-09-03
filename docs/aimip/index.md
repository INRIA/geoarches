# AIMIP: ArchesWeather and ArchesWeatherGen for climate simulations

Here, we show how to run ArchesWeather and ArchesWeatherGen models that were adapted for long climate simulations under the AIMIP protocol.

We provide the pretrained model checkpoints and and code examples.

The models are trained from scratch on 1x1 degree ERA5 data conditioned on monthly forcings (sea surface temperature and sea ice cover) and then rolled out from Oct 1979 to Jan 2025 using prescribed (the same monthly forcings from ERA5).

For details, see the paper: [https://arxiv.org/abs/2605.29976](https://arxiv.org/abs/2605.29976)