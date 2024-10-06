// Function to handle zooming with dynamic step changes
document.getElementById('dot-container').addEventListener('wheel', (e) => {
    e.preventDefault();

    // Dynamically adjust zoom steps based on current zoom level
    let delta;
    if (zoom < 1) {
        delta = e.deltaY > 0 ? -0.05 : 0.05;  // Smaller steps for zoom levels below 1
    } else if (zoom < 5) {
        delta = e.deltaY > 0 ? -0.1 : 0.1;  // Moderate steps for zoom levels between 1 and 5
    } else {
        delta = e.deltaY > 0 ? -0.5 : 0.5;  // Larger steps for zoom levels above 5
    }

    const mouseX = e.clientX - transformContainer.getBoundingClientRect().left; // Adjust for container
    const mouseY = e.clientY - transformContainer.getBoundingClientRect().top;

    const prevZoom = zoom;

    // Update zoom factor
    zoom += delta;

    // Ensure zoom remains between 10% and 5000%, without going negative
    zoom = Math.min(Math.max(0.1, zoom), 50);  // Limit zoom between 0.1 (10%) and 50 (5000%)

    // Calculate the new position to ensure the mouse doesn't move relative to objects
    const newInnerX = (mouseX / prevZoom * zoom) - mouseX;
    const newInnerY = (mouseY / prevZoom * zoom) - mouseY;

    // Update inner position based on zooming
    innerX -= newInnerX;
    innerY -= newInnerY;

    // Apply the transform to the inner content (translation and scaling)
    applyTransformations();
    updateZoomIndicator();
});

// Function to allow manual zoom input by double-clicking the zoom indicator
document.getElementById("zoom-indicator").addEventListener('dblclick', () => {
    const newZoomValue = parseFloat(prompt("Enter desired zoom percentage (10-5000):", zoom * 100));
    if (!isNaN(newZoomValue) && newZoomValue >= 10 && newZoomValue <= 5000) {
        zoom = newZoomValue / 100; // Convert percentage to zoom factor
        applyTransformations();
        updateZoomIndicator();
    } else {
        alert("Invalid zoom value. Please enter a value between 10 and 5000.");
    }
});

// Start panning on middle mouse button down (button code 1)
transformContainer.addEventListener('mousedown', (e) => {
    if (e.button === 1) {  // Only pan when middle mouse button is pressed
        isPanning = true;
        startX = e.clientX - innerX; // Calculate the starting X position
        startY = e.clientY - innerY; // Calculate the starting Y position
    }
});

// Handle mouse movement while panning
transformContainer.addEventListener('mousemove', (e) => {
    if (isPanning) {
        innerX = e.clientX - startX; // Update innerX based on mouse movement
        innerY = e.clientY - startY; // Update innerY based on mouse movement

        // Apply transformation to the transform container
        applyTransformations(); // Apply transformations to the parent container
    }
});

// Stop panning on mouse up
transformContainer.addEventListener('mouseup', () => {
    isPanning = false;
});

// Stop panning if the mouse leaves the transform container
transformContainer.addEventListener('mouseleave', () => {
    isPanning = false;
});