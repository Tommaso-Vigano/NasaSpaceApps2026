/* One local collection fetch; photo stops stay in the map's pixel frame. */
window.createPhotoStops = function (config, canvas) {
    'use strict';
    if (!config.photos_url) return {update() {}};
    const get = (id) => document.getElementById(id);
    const container = get('photo-markers');
    const toggle = get('photos-visible');
    const status = get('photos-status');
    const stopDialog = get('photo-stop');
    const viewer = get('photo-viewer');
    const controller = new AbortController();
    let markers = [], lastView;
    const coordinates = (lat, lon) => '(' + lat.toFixed(5) + ', ' + lon.toFixed(5) + ')';

    function openImage(image) {
        get('photo-full').src = image.image;
        get('photo-full').alt = image.name;
        get('photo-caption').textContent = image.name + ' · ' + coordinates(image.lat, image.lon);
        if (!viewer.open) viewer.showModal();
    }
    function openStop(stop) {
        get('photo-stop-title').textContent = stop.title;
        get('photo-stop-coordinates').textContent = coordinates(stop.lat, stop.lon);
        get('photo-thumbnails').replaceChildren(...stop.images.map((image) => {
            const button = document.createElement('button');
            button.type = 'button'; button.className = 'photo-thumbnail';
            button.title = image.name;
            button.setAttribute('aria-label', 'Open ' + image.name);
            const thumbnail = document.createElement('img');
            thumbnail.src = image.image; thumbnail.alt = image.name;
            button.append(thumbnail);
            button.addEventListener('click', () => openImage(image));
            return button;
        }));
        if (!stopDialog.open) stopDialog.show();
    }
    function update(view) {
        lastView = view;
        if (!view) return;
        const rect = canvas.getBoundingClientRect();
        for (const {stop, button} of markers) {
            const x = (stop.x - view.x) / view.width * rect.width;
            const y = (stop.y - view.y) / view.height * rect.height;
            button.hidden = !toggle.checked || x < 0 || x > rect.width || y < 0 || y > rect.height;
            button.style.left = (rect.left + x) + 'px';
            button.style.top = (rect.top + y) + 'px';
        }
    }
    toggle.addEventListener('change', () => {
        update(lastView);
        if (!toggle.checked) stopDialog.close();
    });
    get('photo-stop-close').addEventListener('click', () => stopDialog.close());
    get('photo-viewer-close').addEventListener('click', () => viewer.close());
    viewer.addEventListener('click', (event) => {
        const rect = viewer.getBoundingClientRect();
        if (event.target === viewer && (event.clientX < rect.left || event.clientX > rect.right
            || event.clientY < rect.top || event.clientY > rect.bottom)) viewer.close();
    });
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && !viewer.open) stopDialog.close();
    });
    viewer.addEventListener('close', () => {
        get('photo-full').removeAttribute('src');
    });
    fetch(config.photos_url, {signal: controller.signal}).then(async (response) => {
        if (!response.ok) throw new Error('Mission photos unavailable.');
        const collection = await response.json();
        markers = collection.stops.map((stop, index) => {
            const button = document.createElement('button');
            button.type = 'button'; button.className = 'photo-marker';
            button.textContent = index + 1;
            button.title = stop.title + ' · ' + coordinates(stop.lat, stop.lon);
            button.setAttribute('aria-label', 'Photos: ' + button.title);
            button.addEventListener('click', () => openStop(stop));
            container.append(button);
            return {stop, button};
        });
        status.textContent = markers.length + ' photo stops';
        toggle.disabled = markers.length === 0;
        update(lastView);
    }).catch((error) => {
        if (error.name !== 'AbortError') {
            status.textContent = error.message;
            toggle.disabled = true;
        }
    });
    window.addEventListener('pagehide', () => controller.abort());
    return {update};
};
