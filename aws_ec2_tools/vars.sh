#!/bin/bash

echo "Please set any relevant varaibles in vars.sh"
exit 1

EXPNAME="sp-test"
COUNT="4"
# TYPE="p3.8xlarge"
# TYPE="p3.2xlarge"
TYPE="g4dn.xlarge"
# AMI="ami-024e767756d8c822a"
AMI="ami-019ef0b24616ab1cd"

KEYNAME="ulma-sjp"
KEYPATH=~/.ssh/ulma-sjp.pem

INSTANCE_FILENAME="aws-$EXPNAME-instanceIds.txt"
