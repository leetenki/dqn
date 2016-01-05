/****************************/
// remote class
/****************************/
var RemoteEnv = function() {
    this.websocket = null;
    this.pixelWidth = 128;
    this.pixelHeight = 96;
    this.waitToSend = false;
    this.prevPixels = new Uint8Array(this.pixelWidth * this.pixelHeight * 4);
    this.nextPixels = new Uint8Array(this.pixelWidth * this.pixelHeight * 4);
    this.sendArray = new Uint8Array(this.pixelWidth * this.pixelHeight * 4 * 2 + 2)
    this.delta = 0.05;
    var container = this;

    this.initConnection = function(url) {
        this.websocket = new WebSocket(url);
        this.websocket.binaryType = "arraybuffer";
        this.websocket.onmessage = this.onmessage;
        return this.websocket;
    }

    this.onmessage = function(e) {
        if(env.getCarSelected().mode == MODE.REMOTE && !container.waitToSend) {
            var car = env.getCarSelected();

            var command = env.getCommandFromAction(parseInt(e.data));
            car.act(command, container.delta);
            env.ui.drawHTML(car);

            container.waitToSend = true;
        }
    }

    // function to send state to server
    this.sendState = function() {
        var car = env.getCarSelected();

        // send texture
        if(remoteMode == "TEXTURE") {
            container.prevPixels.set(container.nextPixels);
            renderer.readRenderTargetPixels(renderTarget, 0, 0, container.pixelWidth, container.pixelHeight, container.nextPixels);

            // create sendArray
            container.sendArray.set(container.prevPixels, 0);
            container.sendArray.set(container.nextPixels, container.prevPixels.length)
            container.sendArray[container.prevPixels.length + container.nextPixels.length] = car.command.action;
            container.sendArray[container.prevPixels.length + container.nextPixels.length + 1] = car.intRewards;

            websocket.send(container.sendArray.buffer)
        } 
        // send distance
        else if(remoteMode == "DISTANCE") {
            var distanceArray = new Array();
            for(var i = 0; i < car.eyes.length; i++) {
                distanceArray.push(Math.round(car.eyes[i].distance / EYE_PARAM.DISTANCE * EYE_PARAM.ROBOT_EYE_DISTANCE));
            }
            var distanceInfo = {
                distance: distanceArray,
                rewards: car.rewards,
                action: car.command.action
            };
            websocket.send(JSON.stringify(distanceInfo));
        }
        container.waitToSend = false;
    }
}