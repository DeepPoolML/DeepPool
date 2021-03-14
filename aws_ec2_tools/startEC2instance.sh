#!/bin/bash
echo "Before using this script, modify EXPNAME in all *.sh files."
echo "For this script, modify EXPNAME & ssh key path."
exit 1

EXPNAME="sp-dev"
INSTANCE_FILENAME="aws-$EXPNAME-instanceIds.txt"
# set -x

totalInstances=`wc -w $INSTANCE_FILENAME | awk '{print $1}'`
if [ "$#" = "1" ]; then
	numInstances="$1"
else
	numInstances="$totalInstances"
fi
startInstanceIndex=$(expr $totalInstances - $numInstances + 1)
instances=`cat $INSTANCE_FILENAME | awk '{print substr($0, index($0, $'"$startInstanceIndex"'))}'`

echo "Starting $numInstances instances of $EXPNAME in $INSTANCE_FILENAME. Instances: $instances"

echo $instances > aws-started-instanceIds.txt
# exit 0

# set -x
OUTOPT=" >> ~/out.log 2>&1"
timestamp=$(date +%s)

#####################################
######### Start instances. ##########
#####################################
aws ec2 start-instances --instance-ids $instances
# sleep 10

dnsnames=`aws ec2 describe-instances --instance-ids $instances --query 'Reservations[].Instances[].PublicDnsName'`
ipaddresses=`aws ec2 describe-instances --instance-ids $instances --query 'Reservations[].Instances[].PublicIpAddress'`

echo $dnsnames > aws-started-publicDnsName.txt
echo $ipaddresses > aws-started-publicIPs.txt

aws ec2 wait instance-status-ok --instance-ids $instances

# sleep 10
##########################################################################################
###### Re-mount volumes (don't format unlike in launchEC2inchance.sh.)
##########################################################################################
# declare -a cmdarr=("sudo mkdir /data" "sudo chown ubuntu:ubuntu /data" "sudo mount -o ro /dev/nvme1n1 /data"
#                    "sudo mkdir /fast" "sudo chown ubuntu:ubuntu /fast" "sudo mkfs -t xfs /dev/nvme0n1" "sudo mount /dev/nvme0n1 /fast"
#                    "cp -r /data/pipedream /fast/")
# declare -a cmdarrMount=("echo 'Re-mounting at $timestamp'" "sudo chown ubuntu:ubuntu /fast" "sudo mkfs -t xfs /dev/nvme0n1" "sudo mount /dev/nvme0n1 /fast")
#                    # "cp -r /data/pipedream /fast/")

for dns in $dnsnames; do
    ssh -i ~/.ssh/ulma-sjp.pem -o StrictHostKeyChecking=no ubuntu@$dns "echo 'Re-started at $timestamp' $OUTOPT"
    echo $dns
    # for cmd in "${cmdarrMount[@]}"; do
    #     echo "  $cmd"
    #     ssh -i ~/.ssh/ulma-sjp.pem -o StrictHostKeyChecking=no ubuntu@$dns "$cmd $OUTOPT"
    #     # sleep 1
    # done
done

##########################################################################################
#### Fetch & upload private IPs. & Hyugens for timetrace
##########################################################################################
privateIps=`aws ec2 describe-instances  --instance-ids $instances --query 'Reservations[].Instances[].PrivateIpAddress'`
echo $privateIps > aws-started-privateIps.txt
ipArr=($privateIps)
i=0
for dns in $dnsnames; do
    scp -i ~/.ssh/ulma-sjp.pem aws-started-privateIps.txt ubuntu@$dns:~/
    scp -i ~/.ssh/ulma-sjp.pem '/Users/seojin/Nextcloud/Research/Huygens (clock sync)/clocksync_prober-1.3-1.x86_64.rpm' ubuntu@$dns:~/

    ssh -i ~/.ssh/ulma-sjp.pem ubuntu@$dns "sudo apt-get --assume-yes install alien"
    ssh -i ~/.ssh/ulma-sjp.pem ubuntu@$dns "sudo alien --scripts clocksync_prober-1.3-1.x86_64.rpm && sudo dpkg -i clocksync-prober_1.3-2_amd64.deb"

    localPrivateIp=${ipArr[$i]}
    ssh -i ~/.ssh/ulma-sjp.pem ubuntu@$dns 'tmux new -d "sudo /opt/clocksync/bin/prober 172.31.72.31 54321 '$localPrivateIp' 54322 '$localPrivateIp' 319 sw |& tee clocksync.log"'
    ((i++))
done

##########################################################################################
###### Prepare pipedream.
##########################################################################################
# declare -a cmdarrPipedream=("sudo rsync -ah /data/docker/ /var/lib/docker/" )
#                             # "echo '$timestamp ## Prepare pipedream'"
# #                             "cd /fast/pipedream; nvidia-docker pull nvcr.io/nvidia/pytorch:19.05-py3"
# #                             "cd /fast/pipedream; docker build --tag pipedream .")
#
#
# for cmd in "${cmdarrPipedream[@]}"; do
#     ssh -i ~/.ssh/ulma-sjp.pem -o StrictHostKeyChecking=no ubuntu@$dns " $OUTOPT"
#     for dns in $dnsnames; do
#         echo $dns
#         echo "  $cmd"
#         ssh -i ~/.ssh/ulma-sjp.pem -o StrictHostKeyChecking=no ubuntu@$dns "$cmd $OUTOPT"
#         # ssh -n -f user@host "sh -c 'cd /whereever; nohup ./whatever > /dev/null 2>&1 &'"
#         sleep 1
#     done
# done

