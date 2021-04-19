#!/bin/bash

source vars.sh

 # aws-$EXPNAME-instanceIds.txt

instances=`aws ec2 describe-instances --filters "Name=tag:Name,Values=$EXPNAME" --query 'Reservations[].Instances[].InstanceId'`
aws ec2 describe-instance-status --instance-ids $instances

#
# while getopts l OPT
# do
#     case "$OPT" in
#         l) SHOW_DETAILS ;;
#     esac
# done
#
# if [ -z "$SHOW_DETAILS" ]; then
# 	aws ec2 describe-instance --instance-ids $INSTANCE
# fi