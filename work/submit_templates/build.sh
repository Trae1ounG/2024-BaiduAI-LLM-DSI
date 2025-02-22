# example: sh build.sh GenRetr_submit

set -x

rm model.zip

zip_path=$1
cd $zip_path
zip -r ../model.zip *
