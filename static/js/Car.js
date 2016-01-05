/************************
// Car object definition
// param.ID
// param.src
// param.mode
// param.size
// param.eyeParam
************************/

var CarMesh = function(param, env) {
	var size = param.size;       // the longest edge
	var offsetY = size / 2;

	var geometry = new THREE.BoxGeometry(size, size, size);
	var texture = THREE.ImageUtils.loadTexture(param.src, undefined, function() {
		if(texture) {
			texture.needsUpdate = true;
			material.needsUpdate = true;
		}
		console.log("car cursor loaded");
	});
	texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;
	var material = new THREE.MeshBasicMaterial({map: texture, side: THREE.DoubleSide});
	var mesh = new THREE.Mesh(geometry, material);
	mesh.position.y = offsetY;
	mesh.objectType = OBJECT_TYPE.CAR;

	return mesh;
}

/**************************
// requires:
// param.ID
// param.size
// param.eyeParam.NUM_EYES
// param.yeParam.COVER
// param.eyeParam.DISTANCE
// param.mode
// param.src
**************************/
var Car = function(param, env) {
	// object variables
	this.mesh = null;
	this.eyes = new Array();
	this.directionVector = new THREE.Vector3(0, 0, 1);
	this.ID = param.ID;
	this.command = COMMAND.FORWARD;
	this.oldWeights = new Array();
	this.layersNeedsUpdate = false;
	this.rewards = 0;
	this.intRewards = 0;
	this.moveSucceeded = true;
	this.brain = initBrain();
	this.size = param.size;       // the longest edge
	this.cameraOffset = this.size / 2;
	this.offsetY = this.size / 2;
	this.mode = param.mode;
	this.eyeGroup = new THREE.Mesh();
	this.eyeGroup.objectType = OBJECT_TYPE.EYE_GROUP;
	this.victim = null;  // used when collision happened
	this.env = env;
	this.world = env.world;
	this.SPEED = param.SPEED;
	this.ROTATE_AMOUNT = param.ROTATE_AMOUNT;
	this.param = param;
	var container = this;

	// define yAxis
	var yAxis = new THREE.Vector3(0, 1, 0);

	// create car mesh
	var geometry = new THREE.BoxGeometry(param.size, param.size, param.size);
	var carTexture = null;
	carTexture = THREE.ImageUtils.loadTexture(param.src, undefined, function() {
		if(carTexture) {
			carTexture.needsUpdate = true;
			container.mesh.material.needsUpdate = true;
		}
		console.log("car loaded");
	});
	carTexture.magFilter = THREE.NearestFilter;
    carTexture.minFilter = THREE.NearestFilter;
	var material = new THREE.MeshBasicMaterial({map: carTexture, side: THREE.DoubleSide, transparent: true});
	this.mesh = new THREE.Mesh(geometry, material);
	this.mesh.position.y = this.offsetY;
	this.mesh.objectType = OBJECT_TYPE.CAR;

	// init old weights
	for(var i = 0; i < 10; i++) {
		var zeroWeights = new Array();
		for(var j = 0; j < param.eyeParam.NUM_EYES; j++) {
			zeroWeights.push(0);
		}
		this.oldWeights.push(zeroWeights);
	}

	/**********************
	//  function to init brain
	**********************/
	function initBrain() {
		var num_inputs = param.eyeParam.NUM_EYES; //  input data size, equals to eyes size.
		var num_actions = COMMAND.LENGTH;         // output size, depends on commands size.
		var network_size = num_inputs + num_actions + num_inputs;

		var layer_defs = [];
		layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});
		layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
		//layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
		layer_defs.push({type:'regression', num_neurons:num_actions});

		// by backpropping the temporal difference learning rule.
		var tdtrainer_options = {learning_rate:0.001, momentum:0.0, batch_size:64, l2_decay:0.01};

		var opt = {};
		opt.temporal_window = 1;
		opt.experience_size = 30000;
		opt.start_learn_threshold = 1000;
		opt.gamma = 0.7;
		opt.learning_steps_total = 200000;
		opt.learning_steps_burnin = 3000;
		opt.epsilon_min = 0.05;
		opt.epsilon_test_time = 0.05;
		opt.layer_defs = layer_defs;
		opt.tdtrainer_options = tdtrainer_options;

		return new deepqlearn.Brain(num_inputs, num_actions, opt); 	
	}
	this.initBrain = initBrain;

	// create eyes
	/*********************
	eye.angleY;     // rotation from y axis.
	eye.target;     // what that eye can see. target mesh or null
	eye.distance;   // distance to target.
	eye.distanceOffset // distance inside box.
	**********************/
	var stepAngle = param.eyeParam.COVER / (param.eyeParam.NUM_EYES - 1);
	var startAngle = -param.eyeParam.COVER / 2;

	if(param.eyeParam.COVER >= Math.PI * 2) {
		stepAngle = param.eyeParam.COVER / (param.eyeParam.NUM_EYES);
		startAngle = (param.eyeParam.COVER - stepAngle) / -2;
	}

	this.mesh.rotationAutoUpdate = true;
	this.mesh.updateMatrix();
	this.mesh.updateMatrixWorld();
	var matrix = new THREE.Matrix4();
	matrix.extractRotation(this.mesh.matrix);
	var direction = new THREE.Vector3(0, 0, 1);
	this.directionVector = direction.applyMatrix4(matrix);
	for(var i = 0; i < param.eyeParam.NUM_EYES; i++) {
		var material = new THREE.LineBasicMaterial({color: param.eyeParam.SAFE_COLOR});
		var geometry = new THREE.Geometry();
		geometry.vertices.push(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1));
		var eye = new THREE.Line(geometry, material);
		eye.angleY = startAngle + stepAngle * i;
		eye.distance = 0;
		this.eyes.push(eye);
		this.eyeGroup.add(eye);

		// compute distance offset
		var ray = new THREE.Raycaster(this.mesh.position, this.directionVector.clone().applyAxisAngle(yAxis, eye.angleY));
		var collisionResults = ray.intersectObjects([this.mesh]);
		eye.distanceOffset = collisionResults[0].distance;
	}

	// function to update eyes
	this.updateEyes = function() {
		// update mesh
		this.mesh.rotationAutoUpdate = true;
		this.mesh.updateMatrix();
		this.mesh.updateMatrixWorld();

		var matrix = new THREE.Matrix4();
		matrix.extractRotation(this.mesh.matrix);
		var direction = new THREE.Vector3(0, 0, 1);
		this.directionVector = direction.applyMatrix4(matrix);

		// get all objects of world
		var allObjects = this.world.getEverythingExceptMe(this.mesh);

		for(var i = 0; i < this.eyes.length; i++) {
			// update each eye
			var ray = new THREE.Raycaster(this.mesh.position, this.directionVector.clone().applyAxisAngle(yAxis, this.eyes[i].angleY));
			var collisionResults = ray.intersectObjects(allObjects);
			if(collisionResults.length > 0 && collisionResults[0].distance < param.eyeParam.DISTANCE) {
				this.eyes[i].geometry.vertices[0].set(this.mesh.position.x, this.mesh.position.y, this.mesh.position.z);
				this.eyes[i].geometry.vertices[1].set(collisionResults[0].point.x, collisionResults[0].point.y, collisionResults[0].point.z)
				this.eyes[i].material.color.set(new THREE.Color(255, collisionResults[0].distance / param.eyeParam.DISTANCE, 0));
				this.eyes[i].target = collisionResults[0].object;
				this.eyes[i].distance = collisionResults[0].distance - this.eyes[i].distanceOffset;
			} else {
				var targetPos = this.mesh.position.clone();
				targetPos.addVectors(targetPos, this.directionVector.clone().applyAxisAngle(yAxis, this.eyes[i].angleY).multiplyScalar(param.eyeParam.DISTANCE))
				this.eyes[i].geometry.vertices[0].set(this.mesh.position.x, this.mesh.position.y, this.mesh.position.z);
				this.eyes[i].geometry.vertices[1].set(targetPos.x, targetPos.y, targetPos.z);
				this.eyes[i].material.color.set(param.eyeParam.SAFE_COLOR);
				this.eyes[i].target = null;
				this.eyes[i].distance = param.eyeParam.DISTANCE - this.eyes[i].distanceOffset;
			}
			this.eyes[i].material.needsUpdate = true;
			this.eyes[i].geometry.verticesNeedUpdate = true;
		}
	}

	// function to move car
	this.move = function(command, delta) {
		var moveSucceeded = true;

		if(command == COMMAND.TURN_RIGHT) {
			this.mesh.rotation.y -= this.ROTATE_AMOUNT * delta;
			if(this.victim = this.world.collisionDetection(this.mesh)) {
				//moveSucceeded = false;
				//this.mesh.rotation.y += this.ROTATE_AMOUNT * delta;
			}
		} else if(command == COMMAND.TURN_LEFT) {
			this.mesh.rotation.y += this.ROTATE_AMOUNT * delta;
			if(this.victime = this.world.collisionDetection(this.mesh)) {
				//moveSucceeded = false;
				//this.mesh.rotation.y -= this.ROTATE_AMOUNT * delta;
			}
		} else if(command == COMMAND.FORWARD) {
			this.mesh.position.addVectors(this.mesh.position.clone(), this.directionVector.clone().multiplyScalar(this.SPEED * delta));
			if(this.victim = this.world.collisionDetection(this.mesh)) {
				console.log(this.victim.objectType.text);
				moveSucceeded = false;
				this.mesh.position.sub(this.directionVector.clone().multiplyScalar(this.SPEED * delta));
			}		
		} else if(command == COMMAND.BACK) {
			this.mesh.position.sub(this.directionVector.clone().multiplyScalar(this.SPEED * delta));
			if(this.victim = this.world.collisionDetection(this.mesh)) {
				moveSucceeded = false;
				this.mesh.position.addVectors(this.mesh.position.clone(), this.directionVector.clone().multiplyScalar(this.SPEED * delta));
			}
		}
		this.updateEyes();
		return moveSucceeded;
	}

	// stop to learn
	this.stopToLearn = function() {
		this.brain.epsilon_test_time = 0.05;
		this.brain.learning = false;
	}

	// start to learn
	this.startToLearn = function() {
		this.brain.epsilon_test_time = 0.05;
		this.brain.learning = true;		
	}

	// switch mode
	this.switchMode = function() {
		switch(this.mode) {
    		case MODE.MANUAL: {
				this.mode = MODE.REMOTE;
    			break;
    		}
    		case MODE.REMOTE: {
				this.mode = MODE.LEARNING;
				this.startToLearn();
    			break;
    		}
    		case MODE.LEARNING: {
				this.mode = MODE.FREEDOM;
				this.stopToLearn();
    			break;
    		}
    		case MODE.FREEDOM: {
				this.mode = MODE.MANUAL;
    			break;
    		}
		}
	}

	// function to think what action to do
	this.think = function() {
		var inputInfo = new Array();
		for(var i = 0; i < this.eyes.length; i++) {
			inputInfo.push(this.eyes[i].distance/param.eyeParam.DISTANCE);
		}
		action = this.brain.forward(inputInfo);
		for(var key in COMMAND) {
			if(COMMAND[key].action == action) {
				this.command = COMMAND[key];
				break;
			}
		}
		return this.command;
	}

	// function to compare two arrays
	this.compareArrays = function(a1, a2) {
		for(var i = 0; i < a1.length; i++) {
			if(a1[i] != a2[i]) {
				return false;
			}
		}
		return true;
	}

	// function to do action
	this.act = function(command, delta) {
		delta *= 3;

		this.command = command;
		this.moveSucceeded = this.move(this.command, delta);

		// compute rewards
		// rewards of distance
		var inDanger = false;
		var distanceRewards = 0;
		var numOfOutsideEyes = 30;
		for(var i = numOfOutsideEyes; i < this.eyes.length - numOfOutsideEyes; i++) {
			var eyeDistance = this.eyes[i].distance / param.eyeParam.DISTANCE;
			distanceRewards += eyeDistance;
			if(eyeDistance < 0.1) {
				inDanger = true;
			}
		}
		distanceRewards /= (this.eyes.length - numOfOutsideEyes*2);

		// rewards of forward
		var forwardRewards = 0;
		if(command == COMMAND.FORWARD && !inDanger && this.moveSucceeded) {
			forwardRewards = 0.2 * distanceRewards;
		} else if(command == COMMAND.BACK) {
			forwardRewards = -0.2;
		}

		// penalty  
		var penalty = 0;
		if(!this.moveSucceeded) {
			penalty = 1;
		}

		// copy current layers to old layers
		// console.log(this.brain.value_net.layers[0].out_act);
		/*
		if(this.brain.value_net.layers[0].out_act) {
			this.layersNeedsUpdate = false;
			if(this.compareArrays(this.oldWeights[0], this.brain.value_net.layers[0].out_act.w)) {
				console.log("same");
			} else {
				console.log("different");
				this.oldWeights = new Array();
				for(var i = 0; i < this.brain.value_net.layers.length; i++) {
					this.oldWeights.push(this.brain.value_net.layers[i].out_act.w)
				}
			}
		}
		*/

		// calc rewards
		this.rewards = distanceRewards + forwardRewards - penalty;
		this.intRewards = Math.ceil((this.rewards + 1) * 128);
		if(this.intRewards > 255) {
			this.intRewards = 255;
		} else if(this.intRewards < 0) {
			this.intRewards = 0;
		}

		// backwards
		this.brain.backward(this.rewards);
	}

	return this;
}

