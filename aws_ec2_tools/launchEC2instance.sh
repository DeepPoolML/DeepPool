#!/bin/bash
# set -x

echo "Before using this script, modify EXPNAME in all *.sh files."
echo "For this script, modify EXPNAME, COUNT, TYPE, AMI, security-group-ids, key-name, subnet accordingly."
exit 1

EXPNAME="sp-dev"
COUNT="4"
# TYPE="p3.8xlarge"
# TYPE="p3.2xlarge"
TYPE="g4dn.xlarge"
# AMI="ami-024e767756d8c822a"
AMI="ami-061ccafb14db56c63"

aws ec2 run-instances \
 --image-id $AMI \
 --security-group-ids sg-02badf64e80ee25af \
 --count $COUNT \
 --instance-type $TYPE \
 --key-name ulma-sjp \
 --subnet-id subnet-e8bd288d \
 --tag-specifications  "ResourceType=instance,Tags=[{Key=Name,Value=$EXPNAME}]"


sleep 60

aws ec2 describe-instances --filters "Name=tag:Name,Values=$EXPNAME" --query 'Reservations[].Instances[].PublicDnsName' > aws-$EXPNAME-publicDnsName.txt
aws ec2 describe-instances --filters "Name=tag:Name,Values=$EXPNAME" --query 'Reservations[].Instances[].InstanceId' > aws-$EXPNAME-instanceIds.txt

instances=`cat aws-$EXPNAME-instanceIds.txt`
dnsnames=`cat aws-$EXPNAME-publicDnsName.txt`

aws ec2 wait instance-status-ok --instance-ids $instances

sleep 10

##########################################################################################
###### Attach dataset volume.
##########################################################################################
# for iid in $instances; do
#     echo $iid
#     echo "aws ec2 attach-volume --device /dev/xvdf —-instance-id $iid —-volume-id vol-0b882fd0b069e7b7f "
#     aws ec2 attach-volume --device /dev/xvdf --instance-id $iid --volume-id vol-0b882fd0b069e7b7f
# done
#
# sleep 15

##########################################################################################
###### Format and mount
##########################################################################################
for dns in $dnsnames; do
    echo $dns
    # echo "ssh -i ~/.ssh/ulma-sjp.pem ubuntu@$dns 'sudo mkdir /data; sudo chown ubuntu:ubuntu /data; sudo mount -o ro /dev/nvme2n1 /data'"
    # ssh -i ~/.ssh/ulma-sjp.pem -o StrictHostKeyChecking=no ubuntu@$dns 'sudo mkdir /data; sudo chown ubuntu:ubuntu /data; sudo mount -o ro /dev/nvme2n1 /data'
    CMD='sudo mkdir /data; sudo chown ubuntu:ubuntu /data'
    #sudo mount -o ro /dev/xvdf /data
    echo "  $CMD"
    ssh -i ~/.ssh/ulma-sjp.pem -o StrictHostKeyChecking=no ubuntu@$dns "$CMD"

    # ssh -i ~/.ssh/ulma-sjp.pem -o StrictHostKeyChecking=no ubuntu@$dns "echo 'UUID=2fa7ec6a-0ae9-4b0a-afcf-e97f27cf9c27  /data  xfs  ro,suid,dev,exec,auto,nouser,async,nofail  0  2' | sudo tee -a /etc/fstab"



    # CMD='sudo mkdir /fast; sudo chown ubuntu:ubuntu /fast; sudo mkfs -t xfs /dev/nvme0n1; sudo mount /dev/nvme0n1 /fast'
    # echo "  $CMD"
    # ssh -i ~/.ssh/ulma-sjp.pem -o StrictHostKeyChecking=no ubuntu@$dns "$CMD"
    #
    # ssh -i ~/.ssh/ulma-sjp.pem -o StrictHostKeyChecking=no ubuntu@$dns "sudo shutdown -r now"
done

sleep 10
aws ec2 wait instance-status-ok \
    --instance-ids $instances

# declare -a cmdarr=("sudo chown ubuntu:ubuntu /fast" "sudo mkfs -t xfs /dev/nvme0n1" "sudo mount /dev/nvme0n1 /fast"
#                   "cp -r /data/pipedream /fast/" "cp -r /data/pipedream ~/")
# for dns in $dnsnames; do
#     echo $dns
#     for cmd in "${cmdarr[@]}"; do
#         echo "  $cmd"
#         ssh -i ~/.ssh/ulma-sjp.pem -o StrictHostKeyChecking=no ubuntu@$dns "$cmd"
#     done
# done

##########################################################################################
###### Upload ssh private key.
##########################################################################################
for dns in $dnsnames; do
    echo $dns
    scp -i ~/.ssh/ulma-sjp.pem ~/.ssh/ulma-sjp.pem ubuntu@$dns:~/.ssh/id_rsa
done