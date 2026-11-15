(() => {
    'use strict';
    const config = JSON.parse(document.getElementById('region-config').textContent);
    const canvas = document.getElementById('region-canvas');
    const context = canvas.getContext('2d');
    const status = document.getElementById('map-status');
    let view, frame = null, drag = null, timer, controller, generation = 0;
    const initial = config.initial_view;
    const endpoints = [];
    let route = [], routeController = null, routing = false;
    const routeStatus = document.getElementById('route-status');
    const routeButton = document.getElementById('calculate-route');
    const clearButton = document.getElementById('clear-endpoints');
    const subdivisionSlider = document.getElementById('subdivisions');
    const mouseCoordinate = document.getElementById('mouse-coordinate');

    function pixelAt(event) {
        const rect = canvas.getBoundingClientRect();
        const x = view.x + (event.clientX-rect.left)/rect.width*view.width;
        const y = view.y + (event.clientY-rect.top)/rect.height*view.height;
        return x >= 0 && y >= 0 && x < config.width && y < config.height ? {x, y} : null;
    }
    function geographic(point) {
        // A bounded metadata grid avoids a server request for every mouse move.
        // The server converts endpoint pixels with the exact CRS for routing.
        const grid = config.coordinate_grid;
        const x = point.x / config.width * (grid[0].length-1);
        const y = point.y / config.height * (grid.length-1);
        const c = Math.min(grid[0].length-2, Math.floor(x));
        const r = Math.min(grid.length-2, Math.floor(y));
        const tx = x-c, ty = y-r;
        return [0, 1].map((axis) =>
            grid[r][c][axis]*(1-tx)*(1-ty) + grid[r][c+1][axis]*tx*(1-ty)
            + grid[r+1][c][axis]*(1-tx)*ty + grid[r+1][c+1][axis]*tx*ty);
    }
    function formatPoint(point) {
        const [lon, lat] = geographic(point);
        return lat.toFixed(5) + ', ' + (((lon+180)%360+360)%360-180).toFixed(5) + ' degrees';
    }
    function updateRouteControls() {
        routeButton.disabled = routing || endpoints.length !== 2;
        clearButton.disabled = routing || endpoints.length === 0;
        subdivisionSlider.disabled = routing;
        const list = document.getElementById('route-endpoints');
        list.replaceChildren(...['Start', 'Destination'].map((name, index) => {
            const item = document.createElement('li');
            item.textContent = name + ': ' + (endpoints[index] ? formatPoint(endpoints[index]) : 'not selected');
            return item;
        }));
    }
    function updateFineness() {
        const amount = Number(subdivisionSlider.value);
        document.getElementById('subdivision-value').textContent = amount;
        const columns = Math.min(config.width-1, amount)+1;
        const rows = Math.min(config.height-1, amount)+1;
        document.getElementById('grid-estimate').textContent =
            'About ' + columns + ' x ' + rows + ' nodes across this sector; capped at native detail.';
        route = [];
        routeStatus.textContent = endpoints.length === 2 ? 'Ready to calculate with this fineness.' : 'Select two points to begin.';
        if (view) draw();
    }
    function drawRouting() {
        const ratio = canvas.width / canvas.getBoundingClientRect().width;
        const screen = (p) => ({x: (p[0]-view.x)/view.width*canvas.width,
                                y: (p[1]-view.y)/view.height*canvas.height});
        context.save();
        if (route.length) {
            const points = route.map(screen);
            context.beginPath(); context.moveTo(points[0].x, points[0].y);
            // Round only a small portion of each corner; authoritative route
            // vertices remain unchanged in source pixels and geographic JSON.
            for (let i = 1; i < points.length-1; i++) {
                const a = points[i-1], b = points[i], c = points[i+1];
                const before = Math.hypot(b.x-a.x, b.y-a.y), after = Math.hypot(c.x-b.x, c.y-b.y);
                const radius = 0.12 * Math.min(before, after);
                if (!before || !after) continue;
                context.lineTo(b.x-(b.x-a.x)/before*radius, b.y-(b.y-a.y)/before*radius);
                context.quadraticCurveTo(b.x, b.y, b.x+(c.x-b.x)/after*radius, b.y+(c.y-b.y)/after*radius);
            }
            const last = points[points.length-1]; context.lineTo(last.x, last.y);
            context.lineJoin = 'round'; context.lineCap = 'round';
            context.lineWidth = 6*ratio; context.strokeStyle = '#18272d'; context.stroke();
            context.lineWidth = 3*ratio; context.strokeStyle = '#a4f5d5'; context.stroke();
        }
        endpoints.forEach((point, index) => {
            const p = screen([point.x, point.y]);
            context.beginPath(); context.arc(p.x, p.y, 7*ratio, 0, Math.PI*2);
            context.fillStyle = index ? '#ffc28a' : '#a4f5d5'; context.fill();
            context.lineWidth = 2*ratio; context.strokeStyle = '#17222a'; context.stroke();
            context.font = 'bold ' + 12*ratio + 'px system-ui';
            context.fillStyle = '#fff'; context.fillText(index ? 'B' : 'A', p.x+11*ratio, p.y-8*ratio);
        });
        context.restore();
    }
    canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    canvas.addEventListener('pointerdown', (event) => {
        if (event.button !== 2) return;
        event.preventDefault();
        if (routing) return;
        const point = pixelAt(event);
        if (!point) { routeStatus.textContent = 'Select a point inside the sector.'; return; }
        if (endpoints.length === 2) endpoints.length = 0;
        endpoints.push(point); route = [];
        routeStatus.textContent = endpoints.length === 2 ? 'Ready. Find route, or right-click to start a new pair.'
            : 'Start selected. Right-click a destination.';
        updateRouteControls(); draw();
    });
    canvas.addEventListener('pointermove', (event) => {
        const point = pixelAt(event);
        mouseCoordinate.textContent = point ? formatPoint(point) : 'Outside sector';
    });
    canvas.addEventListener('pointerleave', () => { mouseCoordinate.textContent = 'Move over the map'; });
    clearButton.addEventListener('click', () => {
        endpoints.length = 0; route = [];
        routeStatus.textContent = 'Select two points to begin.';
        updateRouteControls(); draw();
    });
    subdivisionSlider.addEventListener('input', updateFineness);
    routeButton.addEventListener('click', async () => {
        if (routing || endpoints.length !== 2) return;
        routing = true; route = []; updateRouteControls(); draw();
        routeStatus.textContent = 'Computing route...';
        routeController = new AbortController();
        try {
            const response = await fetch(config.route_url, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, signal: routeController.signal,
                body: JSON.stringify({start: endpoints[0], end: endpoints[1], subdivisions: Number(subdivisionSlider.value)}),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Routing failed.');
            if (!result.path.length) throw new Error('No route was found.');
            route = result.path;
            routeStatus.textContent = result.path.length === 1
                ? 'Both points snap to the same grid cell. Increase subdivisions for more detail.'
                : 'Route found on a ' + result.grid_shape[1] + ' x ' + result.grid_shape[0] + ' grid.';
        } catch (error) {
            if (error.name !== 'AbortError') routeStatus.textContent = error.message;
        } finally {
            routing = false; updateRouteControls(); draw();
        }
    });
    updateFineness(); updateRouteControls();

    function constrain() {
        // Leave a margin when a whole sector is smaller than the viewport.
        view.x = view.width >= config.width ? (config.width - view.width) / 2
            : Math.max(0, Math.min(config.width - view.width, view.x));
        view.y = view.height >= config.height ? (config.height - view.height) / 2
            : Math.max(0, Math.min(config.height - view.height, view.y));
    }
    function draw() {
        context.fillStyle = '#22272e';
        context.fillRect(0, 0, canvas.width, canvas.height);
        if (!frame) { drawRouting(); return; }
        const bounds = frame.bounds;
        context.drawImage(frame.image,
            (bounds.x-view.x)/view.width*canvas.width,
            (bounds.y-view.y)/view.height*canvas.height,
            bounds.width/view.width*canvas.width, bounds.height/view.height*canvas.height);
        drawRouting();
    }
    function schedule() {
        generation++;
        controller?.abort();
        clearTimeout(timer);
        draw();
        timer = setTimeout(load, 150);
    }
    async function load() {
        const version = generation;
        controller = new AbortController();
        const signal = controller.signal;
        const url = new URL(config.frame_url, window.location.origin);
        url.search = new URLSearchParams({...view, pixels: 1024});
        status.textContent = 'Loading visible area...';
        let image;
        try {
            const response = await fetch(url, {signal});
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Could not load map');
            }
            const bounds = JSON.parse(response.headers.get('X-Viewport'));
            image = await createImageBitmap(await response.blob());
            if (version !== generation) { image.close(); return; }
            frame?.image.close();
            frame = {image, bounds};
            draw();
            status.textContent = 'Viewport loaded | ' + (config.width / view.width).toFixed(1)
                + 'x | ' + image.width + ' x ' + image.height + ' image';
        } catch (error) {
            if (error.name !== 'AbortError' && version === generation)
                status.textContent = error.message + '. Move the map or use Fit selection to retry.';
        }
    }
    function fit() {
        const aspect = canvas.width / canvas.height;
        const width = Math.max(initial.width, initial.height * aspect) * 1.05;
        const height = width / aspect;
        view = {x: initial.x + initial.width/2 - width/2,
                y: initial.y + initial.height/2 - height/2, width, height};
        constrain(); schedule();
    }
    function resize() {
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.round(innerWidth * ratio);
        canvas.height = Math.round(innerHeight * ratio);
        if (!view) { fit(); return; }
        const centre = view.y + view.height / 2;
        view.height = view.width * canvas.height / canvas.width;
        view.y = centre - view.height / 2;
        constrain(); schedule();
    }
    function zoom(factor, px = 0.5, py = 0.5) {
        const width = Math.max(8, Math.min(config.width * 2, view.width * factor));
        const height = width * canvas.height / canvas.width;
        view.x += (view.width-width)*px;
        view.y += (view.height-height)*py;
        view.width = width; view.height = height;
        constrain(); schedule();
    }
    canvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        zoom(Math.exp(Math.max(-0.5, Math.min(0.5, event.deltaY*0.0015))),
             (event.clientX-rect.left)/rect.width, (event.clientY-rect.top)/rect.height);
    }, {passive: false});
    canvas.addEventListener('pointerdown', (event) => {
        if (event.button !== 0) return;
        drag = {id: event.pointerId, x: event.clientX, y: event.clientY};
        canvas.setPointerCapture(event.pointerId); canvas.style.cursor = 'grabbing';
    });
    canvas.addEventListener('pointermove', (event) => {
        if (!drag || event.pointerId !== drag.id) return;
        const rect = canvas.getBoundingClientRect();
        view.x -= (event.clientX-drag.x)/rect.width*view.width;
        view.y -= (event.clientY-drag.y)/rect.height*view.height;
        drag.x = event.clientX; drag.y = event.clientY;
        constrain(); schedule();
        const point = pixelAt(event);
        mouseCoordinate.textContent = point ? formatPoint(point) : 'Outside sector';
    });
    function release() { drag = null; canvas.style.cursor = 'grab'; }
    canvas.addEventListener('pointerup', release);
    canvas.addEventListener('pointercancel', release);
    canvas.addEventListener('lostpointercapture', release);
    document.getElementById('zoom-in').addEventListener('click', () => zoom(0.5));
    document.getElementById('zoom-out').addEventListener('click', () => zoom(2));
    document.getElementById('fit-region').addEventListener('click', fit);
    window.addEventListener('resize', resize);
    window.addEventListener('pagehide', () => { controller?.abort(); clearTimeout(timer); frame?.image.close(); routeController?.abort(); });
    resize();
})();
