function setup(){
	noCanvas();
	const model = tf.sequential();
	//model.add(tf.layers.dense({units:1, inputShape:[1]}));
	const values = [];
	tf.tidy(() => {
	for (let i = 0; i <30; i++) {
		values[i] = random(0,100);
}
	const shape = [2,5,3];
	const tensor = tf.tensor3d(values,shape); 
	console.log(tensor);
	});
	

function draw() {
for (let i = 0; i < 1; i++ ){
	const tense1 = tf.scalar(52);}
}

 console.log(tf.memory().numTensors);
	
}