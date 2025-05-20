import './App.css';
import React, { useState } from 'react';
import Tema1 from './teme/Tema1';
import Tema2 from './teme/Tema2';
import Tema3 from './teme/Tema3';
import Tema4 from './teme/Tema4';
import Tema5 from './teme/Tema5';
import Tema6 from './teme/Tema6';
import Tema7 from './teme/Tema7';
import Tema8 from './teme/Tema8';

const teme = [
  { title: "Tema 1", component: <Tema1 /> },
  { title: "Tema 2", component: <Tema2 /> },
  { title: "Tema 3", component: <Tema3 /> },
  { title: "Tema 4", component: <Tema4 /> },
  { title: "Tema 5", component: <Tema5 /> },
  { title: "Tema 6", component: <Tema6 /> },
  { title: "Tema 7", component: <Tema7 /> },
  { title: "Tema 8", component: <Tema8 /> }
];
function App() {
    const [openTema, setOpenTema] = useState(null);

  return (
    <div className="html">
      <div className="body">
        <p class="line-1 anim-typewriter">Teme Calcul Numeric 2025</p>
        <p class="line-2">Medeleanu Daria Vulpescu Bianca A2</p>
        <div className="teme-container">
          {teme.map((tema, idx) => (
            <div key={idx} className="tema-dropdown">
              <button
                className="tema-btn"
                onClick={() => setOpenTema(openTema === idx ? null : idx)}
              >
                {tema.title}
              </button>
              {openTema === idx && (
                <div className="exercise-content" style={{ marginTop: '1em' }}>
                  {tema.component}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
