#!/bin/bash

source vars.sh

OUTOPT=" >> ~/out.log 2>&1"
# OUTOPT=""
timestamp=$(date +%s)

instances=`cat $INSTANCE_FILENAME`

#####################################
######### Stop instances. ##########
#####################################
aws ec2 stop-instances --instance-ids $instances
sleep 10
aws ec2 describe-instances --instance-ids $instances --query 'Reservations[].Instances[].PublicDnsName'
aws ec2 describe-instances --filters "Name=tag:Name,Values=$EXPNAME" --query 'Reservations[].Instances[].PublicDnsName'

# aws ec2 terminate-instances --instance-ids `cat aws-started-instanceIds.txt`