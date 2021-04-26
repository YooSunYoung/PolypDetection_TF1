#num_images=('1' '2' '3' '4' '5' '10' '20' '30' '40' '50' '100' '200' '300' '400' '500' '1000')
num_images=('1')
processor='gpu'
for ((i = 0 ; i < ${#num_images[@]} ; i++)) ; do
  NUM=${num_images[$i]}
  RESULT_FILE='../result/'${NUM}'_image_'${processor}'.txt'
  IMAGE_DIRECTORY='../../data/TimeConsumption/'${NUM}'/'
  echo ${RESULT_FILE}
  echo ${IMAGE_DIRECTORY}
  touch $RESULT_FILE
  for ((j = 0; j < 1; j++)); do
    /home/syo/anaconda3/envs/polyp-tf-gpu/bin/python3 CPU_time_consumption_measurement.py ${RESULT_FILE} &
    echo $NUM' images '${j}'th test'
    sleep 1
    /home/syo/anaconda3/envs/polyp-tf-gpu/bin/python3 image_feeder_cpu.py ${IMAGE_DIRECTORY}
    sleep 1
  done
done

