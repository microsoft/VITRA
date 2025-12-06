window.HELP_IMPROVE_VIDEOJS = false;

// Interpolation feature disabled - not in use
var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 0;

var interp_images = [];
function preloadInterpolationImages() {
  // Disabled - no interpolation images to load
  return;
}

function setInterpolationImage(i) {
  // Disabled - no interpolation images to display
  return;
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }


		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    
    // Only initialize interpolation if the slider exists
    if ($('#interpolation-slider').length > 0) {
      preloadInterpolationImages();
      $('#interpolation-slider').on('input', function(event) {
        setInterpolationImage(this.value);
      });
      setInterpolationImage(0);
      $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);
    }

    bulmaSlider.attach();

})
