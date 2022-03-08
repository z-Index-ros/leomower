isStarted = false;

function enableRover(enable){
    isStarted = enable;
    var enableValue
    if(enable){
        document.getElementById("start-button").style.backgroundColor="#00ff00";
        document.getElementById("stop-button").style.backgroundColor="#ffffff";
    }
    else{
        document.getElementById('stop-button').style.backgroundColor="#ff0000";
        document.getElementById('start-button').style.backgroundColor="#ffffff";
    }
    enableValue  = new ROSLIB.Message({
    data: JSON.stringify(enable)
    });
    onOff.publish(enableValue);
    console.log(enable)
}

function moveRover(direction){
    if(isStarted){
        moveInDir(direction)
    }
}

function moveInDir(direction){
    if(direction == "z"){
        console.log("pressed z")
        var twist = new ROSLIB.Message({
            linear : {
              x : 2,
              y : 0,
              z : 0
            },
            angular : {
              x : 0,
              y : 0,
              z : 0
            }
          });
          cmdVel.publish(twist);
    }
    else if(direction == "q"){
        console.log("pressed q")
        var twist = new ROSLIB.Message({
            linear : {
              x : 0,
              y : 0,
              z : 0
            },
            angular : {
              x : 0,
              y : 0,
              z : 10
            }
          });
          cmdVel.publish(twist);
    }
    else if(direction == "s"){
        console.log("pressed s")
        var twist = new ROSLIB.Message({
            linear : {
              x : -2,
              y : 0,
              z : 0
            },
            angular : {
              x : 0,
              y : 0,
              z : 0
            }
          });
          cmdVel.publish(twist);
    }
    else if(direction == "d"){
        console.log("pressed d")
        var twist = new ROSLIB.Message({
            linear : {
              x : 0,
              y : 0,
              z : 0
            },
            angular : {
              x : 0,
              y : 0,
              z : -10
            }
          });
          cmdVel.publish(twist);
    }
}
