echo="DeepFix raw data download.."
file_id="1GqphtPgxbkaq_tgCtYn4iwpt1vZNdyLL"
file_name="data/deepfix_raw_data/dataset.db"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}

echo="DeepFix preprocessed data download"
file_id="1rOKW8SlhvCiy0ZEgof6sTsY1pUET63DA"
file_name="data/DeepFix.pkl"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}

echo="DrRepair_deepfix preprocessed data download"
file_id="1I4WLTXIjS3ouuD_WIAuTJ2UllyVuf_tG"
file_name="data/DrRepair_deepfix.pkl"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}
