function toggle_ref() {
   document.getElementById('spec').src = 'img/spec_ref.png' ;
   document.getElementById("spec_zoom").style.backgroundImage = "url('img/only_ref.png')";
}

function toggle_inear() {
   document.getElementById('spec').src = 'img/spec_inear.png' ;
   document.getElementById("spec_zoom").style.backgroundImage = "url('img/only_inear.png')";
}

function toggle_hifigan() {
   document.getElementById('spec').src = 'img/spec_hifigan.png' ;
   document.getElementById("spec_zoom").style.backgroundImage = "url('img/only_hifigan.png')";
}

function toggle_kuleshov() {
   document.getElementById('spec').src = 'img/spec_kuleshov.png' ;
   document.getElementById("spec_zoom").style.backgroundImage = "url('img/only_kuleshov.png')";
}

function toggle_seanet() {
   document.getElementById('spec').src = 'img/spec_seanet.png' ;
   document.getElementById("spec_zoom").style.backgroundImage = "url('img/only_seanet.png')";
}

function toggle_eben() {
   document.getElementById('spec').src = 'img/spec_eben.png' ;
   document.getElementById("spec_zoom").style.backgroundImage = "url('img/only_eben.png')";
}

function toggle_streaming() {
   document.getElementById('spec').src = 'img/spec_streaming.png' ;
   document.getElementById("spec_zoom").style.backgroundImage = "url('img/only_streaming.png')";
}


function toggle_intelligibility() {
  var x = document.getElementById("mushra_intelligibility");
  var y = document.getElementById("mushra_quality");

    x.style.display = "block";
    y.style.display = "none";

}

function toggle_quality() {
  var x = document.getElementById("mushra_quality");
  var y = document.getElementById("mushra_intelligibility");

    y.style.display = "none";
    x.style.display = "block";

}

function hide_mushras() {
  var x = document.getElementById("mushra_quality");
  var y = document.getElementById("mushra_intelligibility");

    y.style.display = "none";
    x.style.display = "none";

}
