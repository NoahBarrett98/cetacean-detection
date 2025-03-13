# Dataset

This project is playing around with an update version of the DLDCE 2013 workshop dataset, updated by NOAA to include all Baleen whale calls. To pull  the data set gcloud cli must be installed, use the command: 

```
gsutil -m cp -r \
  "gs://noaa-passive-bioacoustic/dclde/2013/nefsc_sbnms_200903_nopp6_ch10" \
  .
```