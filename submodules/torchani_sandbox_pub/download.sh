#!/bin/bash

# Download the data
echo "Downloading data ..."
wget --no-verbose https://www.dropbox.com/sh/2c8zdqc1hrqsgwy/AAD6l24ngoiFa6DRapF6HPk5a/ -O download.zip
echo "Extracting data ..."
unzip -q download.zip -d download || [[ $? == 2 ]]  # unzip return 2 for dropbox created zip file
