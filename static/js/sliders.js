var slide_z = document.getElementById('z_range'),
    sliderDiv_z = document.getElementById("z_range_slider_amount");

const rangez = document.getElementById('z_range'),
    setValueZ = ()=>{
        const newValue = Number( (rangez.value - rangez.min) * 100 / (rangez.max - rangez.min) ),
            newPosition = 10 - (newValue * 0.2);
        sliderDiv_z.innerHTML = `<span>${rangez.value}</span>`;
        sliderDiv_z.style.left = `calc(${newValue}% + (${newPosition}px))`;
    };
document.addEventListener("DOMContentLoaded", setValueZ);
rangez.addEventListener('input', setValueZ);


slide_z.onchange = function() {
	sliderDiv_z.innerHTML = this.value;
	$.post({
		url: '/slider-control',
		data: {"z_range": this.value},
		success: function(response){
			console.log('new z:', response);
		},
		error: function(error){
			alert(response);
			console.log(error);
		}
	});
}

var slide_net_width = document.getElementById('net_width_range'),
    sliderDiv_net_width = document.getElementById("net_width_range_slider_amount");

const rangenet = document.getElementById('net_width_range'),
    setValueNet = ()=>{
        const newValue = Number( (rangenet.value - rangenet.min) * 100 / (rangenet.max - rangenet.min) ),
            newPosition = 10 - (newValue * 0.2);
        sliderDiv_net_width.innerHTML = `<span>${rangenet.value}</span>`;
        sliderDiv_net_width.style.left = `calc(${newValue}% + (${newPosition}px))`;
    };
document.addEventListener("DOMContentLoaded", setValueNet);
rangenet.addEventListener('input', setValueNet);


slide_net_width.onchange = function() {
  sliderDiv_net_width.innerHTML = this.value;
  $.post({
    url: '/slider-control',
    data: {"net_width_range": this.value},
    success: function(response){
      console.log('new width:', response);
    },
    error: function(error){
      alert(response);
      console.log(error);
    }
  });
}

var slide_z_scale = document.getElementById('z_scale_range'),
    sliderDiv_z_scale = document.getElementById("z_scale_range_slider_amount");

const rangescale = document.getElementById('z_scale_range'),
      rangeV = document.getElementById('rangeVal'),
      setValueScale = ()=>{
          const newValue = Number( (rangescale.value - rangescale.min) * 100 / (rangescale.max - rangescale.min) ),
                newPosition = 10 - (newValue * 0.2);
		sliderDiv_z_scale.innerHTML = `<span>${rangescale.value}</span>`;
		sliderDiv_z_scale.style.left = `calc(${newValue}% + (${newPosition}px))`;
      };
document.addEventListener("DOMContentLoaded", setValueScale);
rangescale.addEventListener('input', setValueScale);

slide_z_scale.onchange = function() {
  sliderDiv_z_scale.innerHTML = this.value;
  $.post({
    url: '/slider-control',
    data: {"z_scale_range": this.value},
    success: function(response){
      console.log('new z scale:', response);
    },
    error: function(error){
      alert(response);
      console.log(error);
    }
  });
}

/*var slide_interpolation = document.getElementById('interpolation_range'),
    sliderDiv_interpolation = document.getElementById("interpolation_slider_amount");

const rangeinter = document.getElementById('interpolation_range'),
    setValueInter = ()=>{
        const newValue = Number( (rangeinter.value - rangeinter.min) * 100 / (rangeinter.max - rangeinter.min) ),
            newPosition = 10 - (newValue * 0.2);
        sliderDiv_interpolation.innerHTML = `<span>${rangeinter.value}</span>`;
        sliderDiv_interpolation.style.left = `calc(${newValue}% + (${newPosition}px))`;
    };
document.addEventListener("DOMContentLoaded", setValueInter);
rangeinter.addEventListener('input', setValueInter);


slide_interpolation.onchange = function() {
    sliderDiv_interpolation.innerHTML = this.value;
    $.post({
        url: '/slider-control',
        data: {"interpolation_range": this.value},
        success: function(response){
            console.log('new interpolation:', response);
        },
        error: function(error){
            alert(response);
            console.log(error);
        }
    });
}
*/
