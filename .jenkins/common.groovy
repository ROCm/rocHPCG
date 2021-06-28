// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()
        
    def command
    def getDependencies = auxiliary.getLibrary('rocPRIM', platform.jenkinsLabel,'develop')
    def compiler = jobName.contains('hipclang') ? '/opt/rocm/bin/hipcc' : '/opt/rocm/bin/hcc'

    command = """
                #!/usr/bin/env bash
                set -x
                ${getDependencies}
                cd ${project.paths.project_build_prefix}
                CXX=${compiler} ${project.paths.build_command}
              """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project)
{
    def command = """
                    #!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    rtest.py -t psdb -i ${project.paths.project_build_prefix}/build/release/tests
                  """

    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/output.xml"
}

return this

