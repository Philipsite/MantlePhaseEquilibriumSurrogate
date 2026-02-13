#=
Script to download necessary files from Zenodo and place them in the appropriate directories.

Warning, this script downloads large files (~2.5 GB). Ensure you have sufficient disk space before running it.
=#
using Downloads
using Tar
using CodecZlib

function download(url::String, dest::String)
    println("Downloading: $url")
    Downloads.download(url, dest;
        progress = (total, now) -> begin
            if total > 0
                pct = round(100 * now / total; digits=1)
                bar_len = 40
                filled = round(Int, bar_len * now / total)
                bar = "█"^filled * "░"^(bar_len - filled)
                print("\r  [$bar] $pct%  ($(round(now/1e6; digits=1)) / $(round(total/1e6; digits=1)) MB)")
            else
                print("\r  Downloaded $(round(now/1e6; digits=1)) MB...")
            end
        end
    )
    println()
end
function fetch_and_extract(url::String, target_dir::String)
    mktempdir() do tmpdir
        tmp_archive = joinpath(tmpdir, "archive.tar.gz")
        download(url, tmp_archive)

        tmp_extract = joinpath(tmpdir, "extracted")
        println("Extracting...")
        open(tmp_archive, "r") do io
            Tar.extract(GzipDecompressorStream(io), tmp_extract)
        end

        # Move extracted contents to target directory
        mkpath(target_dir)
        for item in readdir(tmp_extract)
            src = joinpath(tmp_extract, item)
            dst = joinpath(target_dir, item)
            mv(src, dst; force=true)
        end
        println("Done! Extracted to: $target_dir")
    end
end

data_dir = joinpath(@__DIR__, "data")
models_dir = joinpath(@__DIR__, "models")

if !isdir(data_dir)
    mkpath(data_dir)
end

if !isdir(joinpath(data_dir, "generated_dataset"))
    url = "https://zenodo.org/records/18631853/files/generated_data.tar.gz"
    fetch_and_extract(url, data_dir)
end

if !isdir(joinpath(data_dir, "ood_dataset"))
    url = "https://zenodo.org/records/18631853/files/ood_dataset.tar.gz"
    fetch_and_extract(url, data_dir)
end

if !isdir(models_dir)
    url = "https://zenodo.org/records/18631853/files/models.tar.gz"
    fetch_and_extract(url, models_dir)
end

# This part is commented out, as it is not recommended to re-download the HPT results every time.
# Attention, these results are very large (~10s of GB)!
# if !isdir(joinpath(data_dir, "hpt_results"))
#     url = "https://zenodo.org/records/18631853/files/hpt_results.tar.gz"
#     fetch_and_extract(url, data_dir)
# end
