ranksvm_dir=../../fact-checking--follow-up/experiments/ranksvm
N=$2
out_dir="$1/ranksvm-top-${N}"

echo "N ==> $N"
echo "Out Dir ==> $out_dir"

for ((i=1; i <= 7; i++ )); do 
	echo $N-$i
	echo "RUNNING ==> ./$ranksvm_dir/svm-scale -l -1 -u 1 -s $out_dir/ranksvm.range  $out_dir/train-$N-$i.qid"
	echo "SAVING IN ==>  $out_dir/train.qid.scale"
	time ./$ranksvm_dir/svm-scale -l -1 -u 1 -s $out_dir/ranksvm.range  $out_dir/train-$N-$i.qid > $out_dir/train.qid.scale

	echo "RUNNING ==> ./$ranksvm_dir/svm-train $out_dir/train.qid.scale $out_dir/ranksvm.model"
	time ./$ranksvm_dir/svm-train $out_dir/train.qid.scale $out_dir/ranksvm.model

	echo "RUNNING ==> ./$ranksvm_dir/svm-scale -r $out_dir/ranksvm.range $out_dir/test-$N-$i.qid"
	echo "SAVING IN ==>  $out_dir/test.qid.scale"
	time ./$ranksvm_dir/svm-scale -r $out_dir/ranksvm.range $out_dir/test-$N-$i.qid > $out_dir/test.qid.scale

	echo "RUNNING ==> ./$ranksvm_dir/svm-predict $out_dir/test.qid.scale $out_dir/ranksvm.model $out_dir/test-$N-$i.qid.predict"
	time ./$ranksvm_dir/svm-predict $out_dir/test.qid.scale $out_dir/ranksvm.model $out_dir/test-$N-$i.qid.predict
done