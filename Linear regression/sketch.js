let x_vals = [];
let y_vals = [];
let m,b;  // Creating the slope and bias variables

let learningRate;
let epoch = 0;




function setup(){
	frameRate(60);
	createCanvas(500,500);
	let text1 = createP("Learning Rate:").id("para1");
	let slider1 = createSlider(0,1, 0.01, 0.01).id("slider1");
	let text2 = createP("FrameRate:").id("para2");
	let slider2 = createSlider(1,60, 30 , 1).id("slider2");
	
	let text3 = createP("Epoch:  " + epoch).id("para3");

	text1.html("Learning Rate: " + learningRate);
	
	
	
	
	m = tf.variable(tf.scalar(random(1))); // initialising "m" between 0 and 1 
	b = tf.variable(tf.scalar(random(1))); // initialising "b" 


	}
function mousePressed() {
	//Within square:
	if (mouseX < 500 && mouseY < 500){
		let x = map(mouseX, 0 , width, -1 , 1 );
		let y = map(mouseY, 0 , height, 1 , -1);
		x_vals.push(x);
		y_vals.push(y);
	}	
}

function loss(pred, labels) {
	return pred.sub(labels).square().mean();
}

function predict(arrayinput) {
	const xtensor = tf.tensor1d(arrayinput);
	// y = mx + b;
	const ypred = xtensor.mul(m).add(b);
	return ypred;
}


function draw() {
	let optimizer = tf.train.sgd(learningRate);
	
	
	/// The Sliders!
	learningRate = slider1.value;
	let slider2 = select("#slider2");
	frameRate(slider2.value());
	
	let text1 = select("#para1");
	let text2 = select("#para2");
	let text3 = select("#para3");
	
	text3.html("<b>Epoch:</b> " + epoch);
	text2.html("FrameRate: " + slider2.value());
	text1.html("Learning Rate = " + learningRate);
	/////////////////////////////////
	
	tf.tidy(() => {
	if (x_vals.length > 0) {
	const ys = tf.tensor1d(y_vals);
	optimizer.minimize(() => loss(predict(x_vals),ys));	
	
	}
	
	
	// The Cartesian graph between -1 and 1 with 0 in the middle.
	
	background(0);
	stroke(255);
	strokeWeight(10);
	for (let i= 0; i < x_vals.length; i++) {
		let px = map(x_vals[i], -1 , 1 , 0 , width);
		let py = map(y_vals[i], -1 , 1 , height , 0);
		point(px,py);
	}
	let xpoints = [-1,1];
	let ypoints = predict(xpoints);
		
	let x1 = map(xpoints[0], -1 , 1 , 0 , width);
	let x2 = map(xpoints[1], -1 , 1 , 0 , width);
	
	let liney = ypoints.dataSync();
	
	let y1 = map(liney[0], -1 , 1 , height , 0 );
	let y2 = map(liney[1], -1 , 1 , height , 0 );
	
	strokeWeight(5);
	if (x_vals.length > 1) {	
	line(x1, y1, x2, y2);
	epoch++;
	}
	});
	
}
	
