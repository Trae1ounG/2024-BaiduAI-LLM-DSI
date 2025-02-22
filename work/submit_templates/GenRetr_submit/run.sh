# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# GPU version

input_file=$1

python infer.py \
    --model_name_or_path="./model_checkpoint"\
    --params_path="./model_checkpoint/model_state.pdparams" \
    --test_file "$input_file" \
    --output_path "./predict.txt" \
    --device=gpu \
    --min_dec_len=1\
    --batch_size=64\
    --decode_strategy=beam_search \
    --num_beams=10\
    --num_return_sequences=10 \

