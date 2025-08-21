
# FastSpeech2 + Emotion Label
python3 synthesize.py --text "I'm a boy" \
 --speaker_id 0011 \
 --emotion_id 'Sad' \
 --restore_step 900000 \
 --mode single \
 -p config/ESD/preprocess.yaml -m config/ESD/model.yaml -t config/ESD/train.yaml


# FastSpeech2 + Spherical Emotion Vector
# python3 synthesize.py --use_sphere \
#   --text "I'm a boy" \
#   --emotion_id 'Sad' \
#   --speaker_id 0011 \
#   --restore_step 900000 \
#   --mode single \
#   -p config/ESD/preprocess.yaml -m config/ESD/model.yaml -t config/ESD/train_sphere.yaml

# FastSpeech2 + Spherical Emotion Vector+ Dual Conditional Discriminator
# python3 synthesize.py --use_sphere \
#   --text "I'm a boy" \
#   --emotion_id 'Sad' \
#   --speaker_id 0011  \ 
#   --restore_step 900000 \ # checkpoint
#   --mode single \
#   -p config/ESD/preprocess.yaml -m config/ESD/model.yaml -t config/ESD/train_sphere_adv.yaml # config

