# MARS Explorer
We developed a software to help astronauts and humans in exploring the Mars
surface, using both rovers and human exploration. 

## Cost map

Costs depends on the profile of the explorer, we have different profiles
for humans and rovers.

## Pathfinding
The pathfinding class, is created on a grid cost map (matrix), in which 
each cell contains a float representing a cost. The more the number it's higher
the more the subject is not encouraged to step on that cell.

The matrix is then transformed into a graph, then we add costs on arcs
considering costs that changes considering a particular path (e.g. slopes from 
cell A to B is different from B to A). 

The pathfinding was implemented using the library of NetworkX, useful in
pathfinding using graphs. We used the A* algorithm, which combines efficiency
and can be simply implemented.

## UI
### 3D vision of Mars planet
Mars planet 3d vision, pin that highlights interesting points such as mounts, used landing area 
on previous missions, etc. 
By using the right click you can highlight an area, and it shows a detailed image of the selected area.

### Pathfinding
On the detailed area selected, you can zoom on the image and by using your right
click, by clicking two times you can select a source and a destination to compute the shortest
path.

## NASA data
We used [Mars Trek](https://trek.nasa.gov/mars/#v=0.1&x=0&y=0&z=1&p=urn%3Aogc%3Adef%3Acrs%3AEPSG%3A%3A104905&d=&locale=&b=mars&e=-269.999994963537%2C-128.49609135308955%2C269.999994963537%2C128.49609135308955&sfz=&w=)
 to download different visions of the whole planet. 

We used [USG-NASA Planetary Geologic Mapping program](https://astrogeology-usgs.hub.arcgis.com/pages/interactive-maps-mars)
    to download map of Mars and data.

We used [Curiosity’s Proposed Path up Mount Sharp](https://science.nasa.gov/resource/curiositys-proposed-path-up-mount-sharp/)
to have an idea of a possible path used by a real case on Mars.

We used [Perseverance images](https://mars.nasa.gov/mars2020/multimedia/raw-images/) to help astronauts in recognising their 
while exploring Mars.

## Other sources
- [NASA Science – Perseverance Rover Components](https://science.nasa.gov/mission/mars-2020-perseverance/rover-components/)

- [Rankin A. et al. (2023). Perseverance Rapid Traverse Campaign. IEEE Aerospace Conference](https://ieeexplore.ieee.org/document/10115835/)
- [NASA/JPL – Looking Back at ‘Purgatory Dune’ (PIA07999)](https://science.nasa.gov/photojournal/looking-back-at-purgatory-dune)
- Morgan P. et al. (2018). A pre-landing assessment of regolith properties at the InSight landing site. Space Sci. Rev. 214:104. doi:10.1007/s11214-018-0537-y
- Golombek M., Rapp D. (1997). Size-frequency distributions of rocks on Mars and Earth analog sites. JGR 102(E2):4117–4129. Golombek M. et al. (2021). Rock size-frequency distributions at the InSight landing site. Earth Space Sci. 8:e2021EA001959
- [Arvidson R.E. et al. (2010). Spirit Mars Rover Mission: overview and selected results from the northern Home Plate Winter Haven to the side of Scamander crater. JGR 115. doi:10.1029/2010JE003633](https://www.planetary.org/articles/2204)
- [NASA SP-4214, Where No Man Has Gone Before, ch. 13](https://www.nasa.gov/wp-content/uploads/static/history/SP-4214/ch13-3.html)
- [Jones C.S., Nola F.J. (1971) NASA TM X-64623](https://ntrs.nasa.gov/api/citations/19720004516/downloads/19720004516.pdf)
- [Jacob S.R. et al. (2026). Apollo LRV as-traversed performance metrics. LPSC](https://ntrs.nasa.gov/citations/20250011301)
- [MIT 16.89 Space Systems Engineering (2007), presentation 06](https://ocw.mit.edu/courses/16-89j-space-systems-engineering-spring-2007/68578981bacd42a67b431cad987a6853_presentation_06.pdf)
- [NASA – Space Exploration Vehicle fact sheet](https://www.nasa.gov/wp-content/uploads/2015/08/464826main_sev_factsheet_508.pdf)
- [NASA EVA-EXP-0075, Exploration EVA System ConOps Summary for Artemis Phase 1](https://www.nasa.gov/wp-content/uploads/2020/01/topic_1-_eva_lunar_surface_concept_of_operations.pdf)
- [NASA ICES-2025-294, Step height evaluation (xEMU)](https://ntrs.nasa.gov/api/citations/20250004496/downloads/ICES2025-294-stepheight-final.pdf)
- [NASA ICES 2026, Mars suit mass presentation](https://ntrs.nasa.gov/api/citations/20260005410/downloads/Mars_Mass_ICES2026_Presentation_Final.pdf)
- [NASA ICES 2026, Mars PLSS plan](https://ntrs.nasa.gov/api/citations/20260003024/downloads/Final_ICES2026_Mars_PLSS_Plan.pdf)
- Pandolf K.B., Givoni B., Goldman R.F. (1977). Predicting energy expenditure with loads while standing or walking very slowly. J. Appl. Physiol. 43(4):577–581
- Santee W.R., Allison W.F., Blanchard L.A., Small M.G. (2001). A proposed model for load carriage on sloped terrain. Aviat. Space Environ. Med. 72(6):562–566
- Looney D.P. et al. (2018). Metabolic costs of military load carriage over complex terrain. Mil. Med. 183(9-10):e357
- Richmond P.W., Potter A.W., Santee W.R. (2015). Terrain factors for predicting walking and load carriage energy costs: review and refinement. J. Sport Hum. Perf. 3(3):1–26
- [Kim K.J. et al. (2024). Metabolic workload estimation for EVA traverse. IEEE EMBC](https://ntrs.nasa.gov/api/citations/20240001423/downloads/IEEE%20EMBC%202024.pdf)
- [Parker L. et al. NESC space-weather operational response time (poster)](https://ntrs.nasa.gov/api/citations/20210014156/downloads/Parker%20el%20al.%20Helio%202050%20workshop%20poster_final.pdf)
- Löwe J.L. et al. (2025). Nowcasting solar energetic particle events for Mars missions. arXiv:2502.02469
- [NASA OCHMO-TB-020, Design for Ionizing Radiation Protection (NASA-STD-3001)](https://www.nasa.gov/wp-content/uploads/2023/03/radiation-protection-technical-brief-ochmo.pdf)
- [NASA OCHMO-TB-009, Physiological Effects of Touch Temperature](https://nasa.gov/wp-content/uploads/2023/03/touch-temperature-technical-brief-ochmo.pdf)
- [Mars Climate Database v6.2, LMD/ESA/CNES](https://www-mars.lmd.jussieu.fr/)
- USGS SIM 3464, Geologic map of Jezero crater and the Nili Planum region, Mars (1:75,000).
- Horn B.K.P. (1981). Hill shading and the reflectance map. Proc. IEEE 69(1):14–47 · Riley S.J. et al. (1999). A terrain ruggedness index. Intermountain J. Sci. 5:23–27
