using Documenter
using PartitionedKnetNLPModels


makedocs(
  modules = [PartitionedKnetNLPModels],
  doctest = true,
  # linkcheck = true,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "PartitionedKnetNLPModels.jl",
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(repo = "github.com/paraynaud/PartitionedKnetNLPModels.jl.git", devbranch = "master")
