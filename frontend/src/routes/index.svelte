<script>
    import { Button, Icon } from "sveltestrap";
    import { onMount } from "svelte";
    import FormData from "form-data";

    let fileInput;
    let cameraInput;
    let src;
    let text;
    let displayText;
    let mobileCheck = false;

    // Simple code to detect whether the device is Desktop or Phone
    // Phones/Tablets have additional capability to capture image using phone camera
    onMount(async () => {
        mobileCheck = navigator.userAgent.match(
            /(iPad)|(iPhone)|(android)|(webOS)/i
        );
    });

    const onFileSelected = (e) => {
        src = null;
        text = null;
        displayText = true;

        let image = e.target.files[0];
        let reader = new FileReader();
        reader.readAsDataURL(image);
        reader.onload = (e) => {
            src = e.target.result;
        };
        const dataArray = new FormData();
        dataArray.append("file", image);

        fetch("http://" + location.hostname + ":3001", {
            method: "POST",
            body: dataArray,
        })
            .then((response) => response.json())
            .then(({ data }) => {
                text = data;
                displayText = null;
            })
            .catch((error) => {
                console.log(error);
                console.log("API Call Failed");
            });
    };
</script>

<div id="app">
    <h1 class="text app-title">LABEL YOUR IMAGE</h1>
    <div id="image-parent">
        <img src="border.jpg" alt="border" id="image-border" />
        {#if src}
            <img {src} alt="placeholder" id="image-space" />
        {:else}
            <img src="placeholder.png" alt="placeholder" id="image-space" />
        {/if}
    </div>

    {#if displayText}<p class="text" id="loading-text">Loading...</p>{/if}
    {#if text}
        <ol id="label-list">
            <li class="text">{text[0]}</li>
            <li class="text">{text[1]}</li>
            <li class="text">{text[2]}</li>
        </ol>
    {/if}
    <div>
        {#if mobileCheck}
            <Button
                class="action-button"
                on:click={() => {
                    cameraInput.click();
                }}>Capture Image <Icon name="camera" /></Button
            >
        {/if}
        <Button
            class="action-button"
            on:click={() => {
                fileInput.click();
            }}>Upload Image <Icon name="upload" /></Button
        >
    </div>

    <input
        style="display:none"
        type="file"
        accept="image/*"
        capture
        on:change={(e) => onFileSelected(e)}
        bind:this={cameraInput}
    />
    <input
        style="display:none"
        type="file"
        accept="image/*"
        on:change={(e) => onFileSelected(e)}
        bind:this={fileInput}
    />
</div>

<style>
    #app {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-flow: column;
        margin-top: 2em;
    }

    .text {
        font-family: "Lucida Sans", "Lucida Sans Regular", "Lucida Grande",
            "Lucida Sans Unicode", Geneva, Verdana, sans-serif;
        color: azure;
    }

    .app-title {
        font-family: "Luckiest Guy", cursive;
        font-size: 2.5rem;
        margin: 0.5em;
    }

    #image-parent {
        position: relative;
        top: 0;
        left: 0;
    }

    #image-border {
        position: relative;
        top: 0;
        left: 0;
        height: 450px;
        width: 300px;
        z-index: 0;
    }

    #image-space {
        position: absolute;
        top: 175px;
        left: 25px;
        height: 250px;
        width: 250px;
        z-index: 2;
    }

    #label-list,
    #loading-text {
        margin: 2em;
    }

    :global(.action-button) {
        margin-top: 2em;
        margin-left: 0.5rem;
        margin-right: 0.5rem;
    }
</style>
