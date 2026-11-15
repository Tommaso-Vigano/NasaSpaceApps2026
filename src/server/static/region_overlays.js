/* Shared sharp-cell overlays for the 2D canvas and 3D terrain texture. */
window.drawConstraintLayers = function(context, heatmaps, view, width, height, sectorWidth, sectorHeight) {
        if (!heatmaps) return;
        const mesh = heatmaps.pixel_mesh;
        const nr = mesh.length - 1, nc = mesh[0].length - 1;
        const [rows, cols] = heatmaps.shape;
        const screen = (p) => [(p[0]-view.x)/view.width*width, (p[1]-view.y)/view.height*height];
        // Place native-resolution PNG cells through a small georeferencing
        // mesh. No numeric grid resampling or image smoothing is performed.
        context.save(); context.globalAlpha = .55; context.imageSmoothingEnabled = false;
        context.beginPath();
        context.rect(-view.x/view.width*width, -view.y/view.height*height,
                     sectorWidth/view.width*width, sectorHeight/view.height*height);
        context.clip();
        function triangle(image, source, target) {
            if (Math.max(...target.map(p => p[0])) < 0 || Math.min(...target.map(p => p[0])) > width
                || Math.max(...target.map(p => p[1])) < 0 || Math.min(...target.map(p => p[1])) > height) return;
            const [s0, s1, s2] = source, [p0, p1, p2] = target;
            const ux = s1[0]-s0[0], uy = s1[1]-s0[1], vx = s2[0]-s0[0], vy = s2[1]-s0[1];
            const det = ux*vy-uy*vx;
            const a = ((p1[0]-p0[0])*vy-(p2[0]-p0[0])*uy)/det;
            const c = ((p2[0]-p0[0])*ux-(p1[0]-p0[0])*vx)/det;
            const b = ((p1[1]-p0[1])*vy-(p2[1]-p0[1])*uy)/det;
            const d = ((p2[1]-p0[1])*ux-(p1[1]-p0[1])*vx)/det;
            context.save(); context.beginPath(); context.moveTo(...p0); context.lineTo(...p1); context.lineTo(...p2);
            context.closePath(); context.clip();
            context.transform(a, b, c, d, p0[0]-a*s0[0]-c*s0[1], p0[1]-b*s0[0]-d*s0[1]);
            context.drawImage(image, 0, 0); context.restore();
        }
        heatmaps.layers.filter(layer => layer.visible).forEach((layer) => {
            for (let r = 0; r < nr; r++) for (let c = 0; c < nc; c++) {
                const source = [[c*cols/nc, r*rows/nr], [(c+1)*cols/nc, r*rows/nr],
                                [(c+1)*cols/nc, (r+1)*rows/nr], [c*cols/nc, (r+1)*rows/nr]];
                const target = [mesh[r][c], mesh[r][c+1], mesh[r+1][c+1], mesh[r+1][c]].map(screen);
                for (const indices of [[0, 1, 2], [0, 2, 3]])
                    triangle(layer.image, indices.map(i => source[i]), indices.map(i => target[i]));
            }
        });
        context.restore();
    
};
