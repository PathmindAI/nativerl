PathmindHelper
==============

Introduction
------------

This is the "PathmindHelper" palette item that users can import in their AnyLogic model to make them ready to be uploaded to the Pathmind Web App. It assumes they will get processed by NativeRL. Until the projects are refactored to support sharing the files under `RLlibPolicyHelper`, they need to be kept in sync with the ones from NativeRL.


Required Software
-----------------

* Linux, Mac, or Windows (untested)
* JDK 8+
* [Maven 3+](https://maven.apache.org/download.cgi)


Build Instructions
------------------

 1. Install the JDK and Maven on the system
 2. Run `mvn clean package -Djavacpp.platform.custom -Djavacpp.platform.linux-x86_64 -Djavacpp.platform.macosx-x86_64 -Djavacpp.platform.windows-x86_64` (~231mb)
 3. Inside AnyLogic:
    1. Open the `PathmindPolicyHelper.alp` model
    2. Add RLlibPolicyHelper-0.0.1-SNAPSHOT.jar (generated in step 2) as a dependency.
    3. Click on "Pathmind" in the Projects view
    4. Go to Exporting -> Export the Library -> Finish

By default, this outputs a `PathmindPolicy.jar` file. We can further add to that JAR the files from `RLlibPolicyHelper-0.0.1-SNAPSHOT.jar` to simplify the end user experience, but this also requires modifying the `library.xml` file manually to remove the dependency on the JAR file.


End User Guide
--------------

Currently maintained on [Basecamp](https://3.basecamp.com/3684163/buckets/11875773/messages/2017431518).
