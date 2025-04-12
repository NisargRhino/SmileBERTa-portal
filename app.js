document.getElementById('inputForm').addEventListener('submit', function(event) {
    event.preventDefault();
    document.getElementById('btn-text').style.display = 'none';
    document.getElementById('btn-spinner').style.display = 'inline-block';
    resetStructure();
          
    const smiles1 = document.getElementById('smilesInput1').value;
    //const smiles2 = document.getElementById('smilesInput2').value;
    const protein = document.getElementById('proteinSelector').value;

    if (!protein) {
        alert('Please select a protein.');
        return;
    }
    processCompounds(smiles1, protein);
    
});

async function fetch2DStructure(smiles, imgElement) {
    try {
        const response = await fetch('https://smileberta-portal.onrender.com/get_2d_structure', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ smiles })
        });
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        imgElement.src = url;
    } catch (error) {
        console.error('Error fetching 2D structure:', error);
    }
}

async function fetch3DStructure(smiles, viewerElement, downloadElement, filename) {
    try {
        const response = await fetch('https://smileberta-portal.onrender.com/get_3d_structure', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ smiles })
        });
        const data = await response.json();

        const viewerInstance = $3Dmol.createViewer(viewerElement, {
            defaultcolors: $3Dmol.rasmolElementColors,
            backgroundColor: 'black'
        });

        viewerInstance.addModel(data.pdb, "pdb");
        viewerInstance.setStyle({}, {stick: {colorscheme: 'Jmol'}});
        viewerInstance.zoomTo();
        viewerInstance.render();

        if(downloadElement) {
            downloadElement.style.display = 'block';
            downloadElement.onclick = async function() {
                try {
                    const response = await fetch('https://smileberta-portal.onrender.com/download_pdb', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ smiles, filename })
                    });
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        downloadElement.href = url;
                    } else {
                        throw new Error('Network response was not ok.');
                    }
                } catch (error) {
                    console.error('Error downloading PDB:', error);
                }
            };
        }
        
    } catch (error) {
        console.error('Error fetching 3D structure:', error);
    }
}

async function predictFragment(smiles, suffix, protein) {
    try {
        const response = await fetch('https://smileberta-portal.onrender.com/predict_fragment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ smiles, protein })
        });
        const data = await response.json();
        if (data.error) {
            alert(data.error);
            document.getElementById('btn-text').style.display = 'inline-block';
            document.getElementById('btn-spinner').style.display = 'none';
            return null;
        }

        displayFragmentProperties(suffix, data);
        
        await fetch2DStructure(data.fragment_smiles, document.getElementById(`fragment-2d${suffix}`));
        await fetch3DStructure(data.fragment_smiles, document.getElementById(`viewer-fragmented`), document.getElementById(`fragment-download${suffix}`), `fragment_structure${suffix}.pdb`);
        document.getElementById('btn-text').style.display = 'inline-block';
        document.getElementById('btn-spinner').style.display = 'none';
        //after fetching the 3D structure, scroll to the structure section
        fadeinStructure();
                
        return data.fragment_smiles;
    } catch (error) {
        console.error('Error predicting fragment:', error);
        return null;
    }
}

function displayFragmentProperties(suffix, data) {
    document.getElementById(`fragment-properties${suffix}`).innerHTML = `
            <h3>Properties of Fragment</h3>
            <p style="color:rgb(187, 85, 85);"><strong>SMILES:</strong> ${data.fragment_smiles}</p>
            <p style="color: rgb(187, 85, 85);"><strong>Molecular Weight:</strong> ${data.properties.molecular_weight} Da</p>
            <p style="color: rgb(187, 85, 85);"><strong>LogP Value:</strong> ${data.properties.log_p}</p>
            <p style="color: rgb(187, 85, 85);"><strong>Hydrogen Bond Acceptors:</strong> ${data.properties.hydrogen_bond_acceptors}</p>
            <p style="color: rgb(187, 85, 85);"><strong>Hydrogen Bond Donors:</strong> ${data.properties.hydrogen_bond_donors}</p>
            <p style="color: rgb(187, 85, 85);"><strong>Topological Polar Surface Area:</strong> ${data.properties.tpsa} Å²</p>
        `;
}

async function fetchScore(smiles) {
    try {
        const response = await fetch('https://smileberta-portal.onrender.com/score', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ smiles })
        });
        const data = await response.json();
        return data.score;
    } catch (error) {
        console.error('Error fetching score:', error);
        return null;
    }
}

async function combineFragments(smiles1, smiles2) {

    try {
        const response = await fetch('https://smileberta-portal.onrender.com/combine', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ smiles1, smiles2 })
        });
        const data = await response.json();
        const combinedContainer = document.getElementById('combined-container');
        combinedContainer.innerHTML = ''; // Clear previous results
        
        data.combined_smiles.forEach(element => {
            console.log(element.properties)
        });

        if (data.combined_smiles && data.combined_smiles.length == 0){
            combinedContainer.innerHTML=`
            <p>No possible combined fragments generated</p>
            `
        }else{

            for (let i = 0; i < data.combined_smiles.length; i++) {
                console.log(data.combined_smiles[i])
                const fragment = data.combined_smiles[i]
                // const score = await fetchScore(fragment.smiles);
                const combinedDiv = document.createElement('div');
                combinedDiv.className
                combinedDiv.id = `combined-${i}`;
                combinedDiv.innerHTML = `
                    <h3>Combined Fragment ${i + 1}</h3>
                    <div id="combined-properties-${i}">
                        <p><strong>SMILES:</strong> ${fragment.smiles}</p>
                        <p><strong>Molecular Weight:</strong> ${fragment.properties.molecular_weight} Da</p>
                        <p><strong>LogP Value:</strong> ${fragment.properties.log_p}</p>
                        <p><strong>Hydrogen Bond Acceptors:</strong> ${fragment.properties.hydrogen_bond_acceptors}</p>
                        <p><strong>Hydrogen Bond Donors:</strong> ${fragment.properties.hydrogen_bond_donors}</p>
                        <p><strong>Topological Polar Surface Area:</strong> ${fragment.properties.tpsa} Å²</p>
                    </div>
                    <div class="img-container-combined">
                        <img id="combined-2d-${i}"/>
                    </div>
                    <div id="viewer-combined-${i}">                  
                    </div>
                    <p style="margin-top: 10px; text-align: center;">
                        <a id="combined-download-${i}" 
                        style="background-color: #007bff; color: #ffffff; padding: 8px 16px; border-radius: 4px; text-decoration: none; font-weight: bold; cursor: pointer; display: inline-block;">
                        Download Combined Molecule PDB
                        </a>
                    </p>

                `;
                combinedContainer.appendChild(combinedDiv);
                // Scroll as each fragment is added
                combinedDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });

                // Simulate async processing delay
                await new Promise(resolve => setTimeout(resolve, 500));

                const imgElement = document.getElementById(`combined-2d-${i}`);
                console.log(`Fetching 2D structure for fragment ${i}`);
                await fetch2DStructure(fragment.smiles, imgElement);
                console.log(`2D structure fetched for fragment ${i}`);
            
                // Fetch and render 3D structure
                const viewerElement = document.getElementById(`viewer-combined-${i}`);
                const downloadElement = document.getElementById(`combined-download-${i}`);
                console.log(`Fetching 3D structure for fragment ${i}`);
                await fetch3DStructure(fragment.smiles, viewerElement, downloadElement, `combined_structure_${i}.pdb`);
                console.log(`3D structure fetched for fragment ${i}`);

                // document.getElementById(`toggle-combined-view-${i}`).addEventListener('click', function() {
                //     toggleCombinedView(i);

                // });

            };
        };
    } catch (error) {
        console.error('Error combining fragments:', error);
    }
}

async function processCompounds(smiles1, protein) {
    await fetch2DStructure(smiles1, document.getElementById('input-2d1'));
    await fetch3DStructure(smiles1, document.getElementById('viewer'), document.getElementById('input-download1'), 'input_structure1.pdb');
    await predictFragment(smiles1, "1", protein);
}

function toggleInputView(suffix) {
    const input2D = document.getElementById(`input-2d${suffix}`);
    const viewer = document.getElementById(`viewer${suffix === "1" ? "" : "2"}`);
    const button = document.getElementById(`toggle-input-view${suffix}`);

    if (input2D.style.display === 'none') {
        input2D.style.display = 'block';
        viewer.style.display = 'none';
        button.textContent = 'Show 3D View';
    } else {
        input2D.style.display = 'none';
        viewer.style.display = 'block';
        button.textContent = 'Show 2D View';
    }
}

function toggleFragmentView(suffix) {
    const fragment2D = document.getElementById(`fragment-2d${suffix}`);
    const viewerFragmented = document.getElementById(`viewer-fragmented${suffix === "1" ? "" : "2"}`);
    const button = document.getElementById(`toggle-fragment-view${suffix}`);

    if (fragment2D.style.display === 'none') {
        fragment2D.style.display = 'block';
        viewerFragmented.style.display = 'none';
        button.textContent = 'Show 3D View';
    } else {
        fragment2D.style.display = 'none';
        viewerFragmented.style.display = 'block';
        button.textContent = 'Show 2D View';
    }
}

function toggleCombinedView(index) {
    const combined2D = document.getElementById(`combined-2d${index}`);
    const viewerCombined = document.getElementById(`viewer-combined${index}`);
    //const button = document.getElementById(`toggle-combined-view${index}`);

    if (combined2D.style.display === 'none') {
        combined2D.style.display = 'block';
        viewerCombined.style.display = 'none';
        button.textContent = 'Show 3D View';
    } else {
        combined2D.style.display = 'none';
        viewerCombined.style.display = 'block';
        button.textContent = 'Show 2D View';
    }
}
function fadeOut(element, duration = 500) {
    let opacity = 1;
    const step = 16 / duration;

    function animate() {
        opacity -= step;
        if (opacity <= 0) {
            opacity = 0;
            element.style.opacity = opacity;
            element.style.display = 'none';
        } else {
            element.style.opacity = opacity;
            requestAnimationFrame(animate);
        }
    }

    requestAnimationFrame(animate);
}

function fadeIn(element, duration = 500) {
    element.style.opacity = 0;
    element.style.display = 'block';
    let opacity = 0;
    const step = 16 / duration;

    function animate() {
        opacity += step;
        if (opacity >= 1) {
            opacity = 1;
            element.style.opacity = opacity;
        } else {
            element.style.opacity = opacity;
            requestAnimationFrame(animate);
        }
    }

    requestAnimationFrame(animate);
}

function resetStructure() {
    fadeOut(document.getElementById('structure-section'));
    fadeOut(document.getElementById('add-to-lib'));
    fadeOut(document.getElementById('subtitle-inputted1'));
    fadeOut(document.getElementById('fragment-properties1'));
    fadeOut(document.getElementById('fragment-download1'));
    fadeOut(document.getElementById('toggle-input-view1'));
    fadeOut(document.getElementById('subtitle-fragmented1'));
    fadeOut(document.getElementById('toggle-fragment-view1'));
    document.getElementById('inputForm').scrollIntoView({ behavior: 'smooth' });
}

function fadeinStructure() {
    fadeIn(document.getElementById('structure-section'));
    fadeIn(document.getElementById('add-to-lib'));
    fadeIn(document.getElementById('subtitle-inputted1'));
    fadeIn(document.getElementById('fragment-properties1'));
    fadeIn(document.getElementById('fragment-download1'));
    fadeIn(document.getElementById('toggle-input-view1'));
    fadeIn(document.getElementById('subtitle-fragmented1'));
    fadeIn(document.getElementById('toggle-fragment-view1'));
    document.getElementById('structure-section').scrollIntoView({ behavior: 'smooth' });
}

// Open sidebar and hide the button
toggleCartBtn.addEventListener('click', function() {
    document.getElementById('fragment-cart').classList.add('active');
    toggleCartBtn.style.display = 'none';
});

// Close sidebar and show the button again
document.getElementById('close-cart').addEventListener('click', function() {
    document.getElementById('fragment-cart').classList.remove('active');
    toggleCartBtn.style.display = 'block';
});


