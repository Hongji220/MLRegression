let x_vals = [];
let y_vals = [];
let a, b, c, d; // Creating the slope and bias variables

let learningRate;
let epoch = 0;



function setup() {
	frameRate(60);
	createCanvas(500, 500).id("canvas1");

	//Initialising Sliders:
	createP("Learning Rate:").id("para1");
	createSlider(0, 1, 0.01, 0.01).id("slider1");
	createP("FrameRate:").id("para2");
	createSlider(1, 60, 30, 1).id("slider2");
	createP("Polynomial Degree:").id("para3");
	createP("Epoch:  " + epoch).id("para4");
	createSlider(1, 3, 3, 1).id("slider3");
	


	///////////////////////


	// ax^2 + bx + c :

	a = tf.variable(tf.scalar(random(-1, 1)));
	b = tf.variable(tf.scalar(random(-1, 1)));
	c = tf.variable(tf.scalar(random(-1, 1)));
	d = tf.variable(tf.scalar(random(-1, 1)));


}

function mousePressed() {
	let canvas1 = select("#canvas1");
	//Within square:
	canvas1.mousePressed(function(){
		let x = map(mouseX, 0, width, -1, 1);
		let y = map(mouseY, 0, height, 1, -1);
		x_vals.push(x);
		y_vals.push(y);
		
	})
}


function loss(pred, labels) {
	return pred.sub(labels).square().mean();
}

function predict(arrayinput) {
	const xtensor = tf.tensor1d(arrayinput);
	let degree = select("#slider3");
	let ypred;
	// y = ax + c:
	if (degree.value() == 1) {
		ypred = xtensor.mul(a).add(b);
	} else if (degree.value() == 2) {
		// y = ax^2 + bx + c:
		ypred = xtensor.square().mul(a)
			.add(xtensor.mul(b))
			.add(c);
	} else if (degree.value() == 3) {
		// y = ax^3 + bx^2 + cx + d 
		ypred = xtensor.pow(tf.scalar(3)).mul(a)
			.add(xtensor.pow(tf.scalar(2)).mul(b))
			.add(xtensor.mul(c))
			.add(d);
	}

	return ypred;
}


function draw() {
	let optimizer = tf.train.sgd(learningRate);


	/// The Sliders!
	learningRate = slider1.value;
	let slider2 = select("#slider2");
	let slider3 = select("#slider3");
	frameRate(slider2.value());

	let text1 = select("#para1");
	let text2 = select("#para2");
	let text3 = select("#para3");
	let text4 = select("#para4");

	text4.html("Polynomial Degree: " + slider3.value());
	text3.html("<b>Epoch:</b> " + epoch);
	text2.html("FrameRate: " + slider2.value());
	text1.html("Learning Rate = " + learningRate);
	/////////////////////////////////
	
	 
	tf.tidy(() => {
				if (x_vals.length > 0) {
					const ys = tf.tensor1d(y_vals);
					optimizer.minimize(() => loss(predict(x_vals), ys));

				}


				// The Cartesian graph between -1 and 1 with 0 in the middle.

				background(0);
				stroke(255);
				strokeWeight(10);
				for (let i = 0; i < x_vals.length; i++) {
					let px = map(x_vals[i], -1, 1, 0, width);
					let py = map(y_vals[i], -1, 1, height, 0);
					point(px, py);
				}

				let xpoints = [];
				// creating an array with all the x values from point -1 to 1:
				for (let i = -1; i <= 1.1; i += 0.05) {
					xpoints.push(i);
				}
				// predicting the ypoints from the array of xpoints
				let ypoints = predict(xpoints);

				let liney = ypoints.dataSync();

				// Drawing the curve:
				beginShape();
				noFill();
				stroke(255);
				strokeWeight(5);


				if (x_vals.length > 1) {
					for (let i = 0; i < xpoints.length; i++) {
						let x = map(xpoints[i], -1, 1, 0, width);
						let y = map(liney[i], -1, 1, height, 0);
						vertex(x, y);
						
					}
					epoch++;
				}

				endShape();


			});
			
			}
