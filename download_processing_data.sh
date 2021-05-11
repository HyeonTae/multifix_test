echo="DeepFix raw data download.."
file_id="1GqphtPgxbkaq_tgCtYn4iwpt1vZNdyLL"
file_name="data/deepfix_raw_data/dataset.db"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}

echo="DeepFix preprocessed data download"
file_id="1X5CxD7GRYyL9Mkf8NCQOoILsmzwHmzr1"
file_name="data/DeepFix.tar"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}
tar -xvf data/DeepFix.tar -C ./data
rm data/DeepFix.tar

echo="DrRepair_deepfix preprocessed data download"
file_id="1MjCtVrL4l7_SQOkkVayx2G6MY0JOwuVV"
file_name="data/DrRepair_deepfix.tar"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}
tar -xvf data/DrRepair_deepfix.tar -C ./data
rm data/DrRepair_deepfix.tar
