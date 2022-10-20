// use your mousewheel to zoom in 🔍

console.clear();

const image = document.querySelectorAll('.image')[0];
const zoom = document.querySelectorAll('.zoom')[0];
const zoomImage = document.querySelectorAll('.zoom-image')[0];

let clearSrc;
let zoomLevel = 1;

const images = [
	{
		thumb: 'https://images.unsplash.com/photo-1480796927426-f609979314bd?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',
		hires: 'https://images.unsplash.com/photo-1480796927426-f609979314bd?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2560&q=80'
	}, {
		thumb: 'https://images.unsplash.com/photo-1503899036084-c55cdd92da26?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',
		hires: 'https://images.unsplash.com/photo-1503899036084-c55cdd92da26?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2560&q=80'
	}, {
		thumb: 'https://images.unsplash.com/photo-1490761668535-35497054764d?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',
		hires: 'https://images.unsplash.com/photo-1490761668535-35497054764d?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2560&q=80'
	}, {
		thumb: 'https://images.unsplash.com/photo-1565175508370-0fff04b6bb5a?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',
		hires: 'https://images.unsplash.com/photo-1565175508370-0fff04b6bb5a?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2560&q=80'
	}, {
		thumb: 'https://images.unsplash.com/photo-1522547902298-51566e4fb383?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',
		hires: 'https://images.unsplash.com/photo-1522547902298-51566e4fb383?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2560&q=80'
	}, 
]

// set to random image
let img = images[Math.floor(Math.random() * images.length)];

image.getElementsByTagName('a')[0].setAttribute('href', img.hires);
image.getElementsByTagName('img')[0].setAttribute('src', img.thumb);

const preloadImage = url => {
	let img = new Image();
	img.src = url;
}

preloadImage(img.hires);



const enterImage = function(e) {
	zoom.classList.add('show', 'loading');
	clearTimeout(clearSrc);
	
	let posX, posY, touch = false;
	
	if (e.touches) {
		posX = e.touches[0].clientX;
		posY = e.touches[0].clientY;
		touch = true;
	} else {
		posX = e.clientX;
		posY = e.clientY;
	}
	
	touch
		? zoom.style.top = `${posY - zoom.offsetHeight / 1.25}px`
		: zoom.style.top = `${posY - zoom.offsetHeight / 2}px`;
	zoom.style.left = `${posX - zoom.offsetWidth / 2}px`;
	
	let originalImage = this.getElementsByTagName('a')[0].getAttribute('href');
	
	zoomImage.setAttribute('src', originalImage);
	
	// remove the loading class
	zoomImage.onload = function() {
		console.log('hires image loaded!');
		setTimeout(() => {
			zoom.classList.remove('loading');
		}, 500);
	}
}


const leaveImage = function() {
	// remove scaling to prevent non-transition 
	zoom.style.transform = null;
	zoomLevel = 1;
	zoom.classList.remove('show');
	clearSrc = setTimeout(() => {
							 zoomImage.setAttribute('src', '');
						 }, 250);
}


const move = function(e) {
	e.preventDefault();
	
	let posX, posY, touch = false;
	
	if (e.touches) {
		posX = e.touches[0].clientX;
		posY = e.touches[0].clientY;
		touch = true;
	} else {
		posX = e.clientX;
		posY = e.clientY;
	}
	
	// move the zoom a little bit up on mobile (because of your fat fingers :<)
	touch
		? zoom.style.top = `${posY - zoom.offsetHeight / 1.25}px`
		: zoom.style.top = `${posY - zoom.offsetHeight / 2}px`;
	zoom.style.left = `${posX - zoom.offsetWidth / 2}px`;
	
	let percX = (posX - this.offsetLeft) / this.offsetWidth,
			percY = (posY - this.offsetTop) / this.offsetHeight;
	
	let zoomLeft = -percX * zoomImage.offsetWidth + (zoom.offsetWidth / 2),
			zoomTop = -percY * zoomImage.offsetHeight + (zoom.offsetHeight / 2);
	
	zoomImage.style.left = `${zoomLeft}px`;
	zoomImage.style.top = `${zoomTop}px`;
}



image.addEventListener('mouseover', enterImage);
image.addEventListener('touchstart', enterImage);

image.addEventListener('mouseout', leaveImage);
image.addEventListener('touchend', leaveImage);

image.addEventListener('mousemove', move);
image.addEventListener('touchmove', move);


image.addEventListener('wheel', e => {
	e.preventDefault();
	e.deltaY > 0 ? zoomLevel-- : zoomLevel++;
	
	if (zoomLevel < 1) zoomLevel = 1;
	if (zoomLevel > 5) zoomLevel = 5;
	
	console.log(`zoom level: ${zoomLevel}`);
	zoom.style.transform = `scale(${zoomLevel})`;
});